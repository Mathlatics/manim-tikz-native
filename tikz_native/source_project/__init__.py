"""Public source-project API with hardened generated-source rewriting.

The implementation file remains importable as an internal module while this
package supplies the public import path used by the console entry point.  This
keeps the repair isolated from unrelated Provider modules and allows the
rewriter to fail closed without executing generated code.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Mapping

_IMPLEMENTATION_PATH = Path(__file__).resolve().parent.parent / "source_project.py"
_SPEC = importlib.util.spec_from_file_location(
    "tikz_native._source_project_implementation", _IMPLEMENTATION_PATH
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - installation guard
    raise ImportError(f"cannot load source-project implementation: {_IMPLEMENTATION_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _matching_closing_parenthesis(source: str, opening: int) -> int:
    depth = 0
    quote: str | None = None
    triple = False
    escaped = False
    comment = False
    index = opening
    while index < len(source):
        char = source[index]
        if comment:
            if char == "\n":
                comment = False
            index += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
                index += 1
                continue
            if char == "\\":
                escaped = True
                index += 1
                continue
            if triple:
                if source.startswith(quote * 3, index):
                    quote = None
                    triple = False
                    index += 3
                else:
                    index += 1
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char == "#":
            comment = True
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            triple = source.startswith(char * 3, index)
            index += 3 if triple else 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise _IMPL.SourceProjectBuildError("unterminated OpenFaceOcclusion3D call")


def _insert_open_face_arguments(
    source: str,
    *,
    paint_policy: str,
    painter_z_band: Any,
) -> str:
    call_pattern = re.compile(r"OpenFaceOcclusion3D\s*\(")
    if call_pattern.search(source) is None:
        if re.search(r"open[_ -]?face|OpenFace", source, flags=re.IGNORECASE):
            raise _IMPL.SourceProjectBuildError(
                "generated source contains an open-face implementation but does not "
                "expose the current OpenFaceOcclusion3D binding"
            )
        return source

    binding_patterns = (
        r"\bfrom\s+[\w.]+\s+import\s+[^\n]*\bOpenFaceOcclusion3D\b",
        r"\bOpenFaceOcclusion3D\s*=",
        r"\bclass\s+OpenFaceOcclusion3D\b",
    )
    if not any(re.search(pattern, source) for pattern in binding_patterns):
        raise _IMPL.SourceProjectBuildError(
            "generated source calls OpenFaceOcclusion3D without exposing the current binding"
        )

    source = re.sub(
        r"compositing_mode\s*=\s*(['\"])[^'\"]*\1",
        'compositing_mode="unified"',
        source,
    )
    arguments = {
        "compositing_mode": 'compositing_mode="unified"',
        "paint_policy": f"paint_policy={_IMPL.project_literal(paint_policy)}",
        "painter_z_band": (
            "painter_z_band="
            + _IMPL.project_literal(tuple(painter_z_band.as_list()))
        ),
    }

    for match in reversed(list(call_pattern.finditer(source))):
        opening = match.end() - 1
        closing = _matching_closing_parenthesis(source, opening)
        body = source[opening + 1 : closing]
        missing = [
            argument
            for name, argument in arguments.items()
            if re.search(rf"\b{name}\s*=", body) is None
        ]
        if not missing:
            continue
        stripped = body.rstrip()
        trailing_space = body[len(stripped) :]
        replacement = (
            stripped + ", " + ", ".join(missing)
            if stripped
            else ", ".join(missing)
        ) + trailing_space
        source = source[: opening + 1] + replacement + source[closing:]
    return source


_ORIGINAL_BRIDGE_GENERATOR = _IMPL._call_bridge_generator


def _safe_bridge_generator(request: Mapping[str, Any]) -> str | None:
    # A Bridge response envelope containing generated Python is already a
    # derived result.  Sending it back through the request API can misinterpret
    # it as an authored request, so let the caller extract it directly.
    if _IMPL._extract_bridge_python(request) is not None:
        return None
    return _ORIGINAL_BRIDGE_GENERATOR(request)


_IMPL._insert_open_face_arguments = _insert_open_face_arguments
_IMPL._call_bridge_generator = _safe_bridge_generator

for _name in _IMPL.__all__:
    globals()[_name] = getattr(_IMPL, _name)

__all__ = list(_IMPL.__all__)
