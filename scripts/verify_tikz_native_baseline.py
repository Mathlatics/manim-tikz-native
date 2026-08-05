#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tikz_native import compile_document  # noqa: E402
from tikz_native.regression import (  # noqa: E402
    build_semantic_snapshot,
    sha256_file,
)


DEFAULT_BASELINE = (
    ROOT
    / "reports"
    / "tikz_native"
    / "2026_national_1_18"
    / "baseline-v0.1.json"
)


def _compare(expected: Any, actual: Any, path: str, issues: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            issues.append(f"{path}: expected object, got {type(actual).__name__}")
            return
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys):
            issues.append(f"{path}.{key}: missing")
        for key in sorted(actual_keys - expected_keys):
            issues.append(f"{path}.{key}: unexpected")
        for key in sorted(expected_keys & actual_keys):
            _compare(expected[key], actual[key], f"{path}.{key}", issues)
        return

    if isinstance(expected, list):
        if not isinstance(actual, list):
            issues.append(f"{path}: expected list, got {type(actual).__name__}")
            return
        if len(expected) != len(actual):
            issues.append(
                f"{path}: expected {len(expected)} items, got {len(actual)}"
            )
            return
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=True)
        ):
            _compare(expected_item, actual_item, f"{path}[{index}]", issues)
        return

    if expected != actual:
        issues.append(f"{path}: expected {expected!r}, got {actual!r}")


def _audit_native_sources(paths: list[Path]) -> list[str]:
    forbidden = {"VMobject", "SVGMobject", "ImageMobject", "set_points_as_corners"}
    issues: list[str] = []
    for path in paths:
        if not path.exists():
            issues.append(f"native audit source missing: {path}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                continue
            if name in forbidden:
                issues.append(f"{path}:{node.lineno}: forbidden call {name}(...)")
    return issues


def verify_baseline(
    baseline_path: Path,
    source_override: Path | None = None,
    *,
    verify_evidence: bool = True,
) -> list[str]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    source = source_override or Path(baseline["source"]["path"])
    issues: list[str] = []

    if not source.exists():
        return [f"source missing: {source}"]

    document = compile_document(source)
    if document.source_sha256 != baseline["source"]["sha256"]:
        issues.append(
            "source.sha256: expected "
            f"{baseline['source']['sha256']}, got {document.source_sha256}"
        )

    actual_semantic = build_semantic_snapshot(document)
    _compare(baseline["semantic"], actual_semantic, "semantic", issues)

    native_paths = [ROOT / path for path in baseline["native_audit_paths"]]
    issues.extend(_audit_native_sources(native_paths))

    if verify_evidence:
        evidence_root = baseline_path.parent
        for relative_path, expected in baseline["evidence_files"].items():
            path = evidence_root / relative_path
            if not path.exists():
                issues.append(f"evidence missing: {relative_path}")
                continue
            actual_size = path.stat().st_size
            if actual_size != expected["bytes"]:
                issues.append(
                    f"evidence {relative_path}.bytes: expected "
                    f"{expected['bytes']}, got {actual_size}"
                )
            actual_sha256 = sha256_file(path)
            if actual_sha256 != expected["sha256"]:
                issues.append(
                    f"evidence {relative_path}.sha256: expected "
                    f"{expected['sha256']}, got {actual_sha256}"
                )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the frozen TikZ-to-native-Manim v0.1 baseline."
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--input",
        type=Path,
        help="Override the source path while retaining the frozen source hash.",
    )
    parser.add_argument(
        "--skip-evidence",
        action="store_true",
        help="Check semantics and native policy without hashing render evidence.",
    )
    args = parser.parse_args()

    issues = verify_baseline(
        args.baseline,
        args.input,
        verify_evidence=not args.skip_evidence,
    )
    if issues:
        print("TikZ → 原生 Manim v0.1 基线：未通过")
        for issue in issues:
            print(f"- {issue}")
        return 1

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    print("TikZ → 原生 Manim v0.1 基线：通过")
    print(f"- 源文件 SHA-256：{baseline['source']['sha256']}")
    print(
        f"- 语义对象：{baseline['semantic']['picture_count']} 图 / "
        f"{baseline['semantic']['object_count']} 对象"
    )
    print(f"- 渲染证据：{len(baseline['evidence_files'])} 项完整")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
