from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .macro_frontend import MacroFrontendError, materialize_entry_macro
from .projection_3d import (
    Basis2,
    Matrix3,
    matrix_from_tikz_basis,
    tikz_three_d_view_basis,
)
from .occlusion_3d import OcclusionGeometryError, parallel_occlusion_interval


TEX_PT_PER_CM = 72.27 / 2.54
MM_TO_PT = 72.27 / 25.4
HINGE_RELATION_SCHEMA = "tikz-native-hinge-relation/v1"


class TikzNativeError(RuntimeError):
    """Raised when the restricted compiler cannot safely interpret input."""


@dataclass(frozen=True)
class Length:
    pt: float

    def to_json(self) -> dict[str, float]:
        return {"pt": self.pt}


@dataclass
class StyleSpec:
    draw_color: str | None = "#000000"
    fill_color: str | None = None
    opacity: float = 1.0
    fill_opacity: float | None = None
    draw_opacity: float | None = None
    line_width_pt: float = 0.9
    line_cap: str = "round"
    line_join: str = "round"
    dash_pattern_pt: tuple[float, float] | None = None
    arrow_tip: str | None = None
    arrow_length_pt: float | None = None
    arrow_width_pt: float | None = None
    font_command: str | None = None
    inner_xsep_pt: float | None = None
    inner_ysep_pt: float | None = None
    text_color: str | None = None
    node_border_color: str | None = None
    transform_shape: bool = False
    native_canvas_plane: str | None = None
    rectangle_node: bool = False
    rotate_degrees: float = 0.0
    raw_options: list[str] = field(default_factory=list)


@dataclass
class LabelPlacement:
    anchor: str = "center"
    dx_pt: float = 0.0
    dy_pt: float = 0.0
    path_pos: float | None = None
    sloped: bool = False
    font_command: str | None = None


@dataclass
class ObjectSpec:
    id: str
    kind: str
    geometry: dict[str, Any]
    style: StyleSpec
    z_index: int
    source_line: int
    raw: str
    label: str | None = None
    placement: LabelPlacement | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class OcclusionRelationSpec:
    id: str
    start_name: str
    end_name: str
    face_names: list[str]
    visible_style: StyleSpec
    hidden_style: StyleSpec
    object_ids: list[str]
    z_index: int
    source_line: int
    raw: str


@dataclass
class HingeRelationSpec:
    """One explicit, non-drawing 3D hinge authorship relation.

    The directed axis owns the sign of future rotations.  Fixed and moving
    faces use coordinate identities rather than generated object IDs so the
    relation survives harmless changes to drawable object naming.
    """

    id: str
    axis_names: list[str]
    fixed_face_names: list[str]
    moving_face_names: list[str]
    source_line: int
    raw: str
    schema: str = HINGE_RELATION_SCHEMA


@dataclass
class NamedPathSpec:
    name: str
    kind: str
    geometry: dict[str, Any]
    source_line: int
    raw: str


@dataclass
class IntersectionSpec:
    path_a: str
    path_b: str
    sort_by: str
    coordinate_names: list[str]
    points: list[tuple[float, float]]
    sort_parameters: list[float]
    source_line: int
    raw: str


@dataclass
class Projection3DSpec:
    source: str
    matrix: Matrix3
    x_basis_cm: Basis2
    y_basis_cm: Basis2
    z_basis_cm: Basis2
    azimuth_degrees: float | None = None
    elevation_degrees: float | None = None


@dataclass
class PictureSpec:
    index: int
    start_line: int
    end_line: int
    raw_options: str
    scale: float
    line_width_pt: float
    line_cap: str
    line_join: str
    dimension: int = 2
    projection_3d: Projection3DSpec | None = None
    named_styles: dict[str, str] = field(default_factory=dict)
    coordinates: dict[str, tuple[float, ...]] = field(default_factory=dict)
    coordinate_dependencies: dict[str, dict[str, Any]] = field(default_factory=dict)
    symbols: dict[str, float | Length] = field(default_factory=dict)
    named_paths: dict[str, NamedPathSpec] = field(default_factory=dict)
    intersections: list[IntersectionSpec] = field(default_factory=list)
    objects: list[ObjectSpec] = field(default_factory=list)
    occlusion_relations: list[OcclusionRelationSpec] = field(default_factory=list)
    hinge_relations: list[HingeRelationSpec] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    animation_steps: list[dict[str, str]] = field(default_factory=list)


@dataclass
class DocumentSpec:
    source_path: str
    source_sha256: str
    colors: dict[str, str]
    pictures: list[PictureSpec]
    entry_macro: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for picture in payload["pictures"]:
            for key, value in list(picture["symbols"].items()):
                if isinstance(value, dict) and "pt" in value:
                    continue
                original = next(
                    p for p in self.pictures if p.index == picture["index"]
                ).symbols[key]
                if isinstance(original, Length):
                    picture["symbols"][key] = original.to_json()
        return payload

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


@dataclass(frozen=True)
class _PictureSource:
    index: int
    start_line: int
    end_line: int
    options: str
    body: str
    prelude: str = ""


@dataclass(frozen=True)
class _CoordValue:
    xy: tuple[float, ...]
    name: str | None = None
    dependency: dict[str, Any] | None = None


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _strip_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        escaped = False
        kept: list[str] = []
        for char in line:
            if char == "%" and not escaped:
                break
            kept.append(char)
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
        lines.append("".join(kept))
    return "\n".join(lines)


def _extract_balanced(
    text: str,
    start: int,
    opener: str = "{",
    closer: str = "}",
) -> tuple[str, int]:
    if start >= len(text) or text[start] != opener:
        raise TikzNativeError(f"Expected {opener!r} at offset {start}")
    depth = 0
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    raise TikzNativeError(f"Unbalanced {opener}{closer} starting at {start}")


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    buffer: list[str] = []
    stack: list[str] = []
    pairs = {"{": "}", "[": "]", "(": ")"}
    closers = set(pairs.values())
    escaped = False
    for char in text:
        if escaped:
            buffer.append(char)
            escaped = False
            continue
        if char == "\\":
            buffer.append(char)
            escaped = True
            continue
        if char in pairs:
            stack.append(pairs[char])
        elif char in closers and stack and char == stack[-1]:
            stack.pop()
        if char == delimiter and not stack:
            part = "".join(buffer).strip()
            if part:
                parts.append(part)
            buffer = []
        else:
            buffer.append(char)
    part = "".join(buffer).strip()
    if part:
        parts.append(part)
    return parts


def _slug(text: str) -> str:
    cleaned = re.sub(r"\\(?:left|right|small|quad|,|;|!)", "", text)
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", cleaned).strip("_")
    return cleaned[:32] or "label"


class _SafeExpressionEvaluator(ast.NodeVisitor):
    _binary = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
    }
    _unary = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}
    _functions = {
        "sqrt": math.sqrt,
        "abs": abs,
        "min": min,
        "max": max,
        # PGF math trigonometric functions take degrees, not radians.
        "sin": lambda value: math.sin(math.radians(value)),
        "cos": lambda value: math.cos(math.radians(value)),
        "tan": lambda value: math.tan(math.radians(value)),
    }

    def visit_Expression(self, node: ast.Expression) -> float:
        return float(self.visit(node.body))

    def visit_Constant(self, node: ast.Constant) -> float:
        if not isinstance(node.value, (int, float)):
            raise TikzNativeError(f"Unsupported expression constant: {node.value!r}")
        return float(node.value)

    def visit_BinOp(self, node: ast.BinOp) -> float:
        operation = self._binary.get(type(node.op))
        if operation is None:
            raise TikzNativeError(f"Unsupported operator: {type(node.op).__name__}")
        return float(operation(self.visit(node.left), self.visit(node.right)))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operation = self._unary.get(type(node.op))
        if operation is None:
            raise TikzNativeError(f"Unsupported unary operator: {type(node.op).__name__}")
        return float(operation(self.visit(node.operand)))

    def visit_Call(self, node: ast.Call) -> float:
        if not isinstance(node.func, ast.Name) or node.func.id not in self._functions:
            raise TikzNativeError("Unsupported function call in PGF math expression")
        if node.keywords:
            raise TikzNativeError("Keyword arguments are not supported in PGF math")
        return float(self._functions[node.func.id](*(self.visit(arg) for arg in node.args)))

    def visit_Name(self, node: ast.Name) -> float:
        if node.id == "pi":
            return math.pi
        raise TikzNativeError(f"Unknown expression name: {node.id}")

    def generic_visit(self, node: ast.AST) -> float:
        raise TikzNativeError(f"Unsupported PGF expression node: {type(node).__name__}")


class TikzNativeCompiler:
    """Compile a deliberate subset of TikZ into semantic native-object specs."""

    _direction_keys = {
        "above",
        "below",
        "left",
        "right",
        "above left",
        "above right",
        "below left",
        "below right",
    }
    _anchor_map = {
        "center": "center",
        "north": "north",
        "south": "south",
        "west": "west",
        "east": "east",
        "north west": "north west",
        "north east": "north east",
        "south west": "south west",
        "south east": "south east",
    }

    def __init__(
        self,
        source_path: str | Path | None = None,
        *,
        source_text: str | None = None,
        entry_macro: str | None = None,
    ):
        if (source_path is None) == (source_text is None):
            raise TikzNativeError("Pass exactly one of source_path or source_text")
        if source_text is not None and not isinstance(source_text, str):
            raise TikzNativeError("source_text must be a string")
        self.source_path = (
            Path(source_path).expanduser().resolve()
            if source_path is not None
            else None
        )
        self.source_text = (
            source_text
            if source_text is not None
            else self.source_path.read_text(encoding="utf-8")
        )
        self.clean_text = _strip_comments(self.source_text)
        self.colors = self._extract_colors(self.clean_text)
        self.macros = self._extract_zero_arg_gdefs(self.clean_text)
        self.entry_macro = entry_macro.strip().lstrip("\\") if entry_macro else None
        self._object_ids: dict[str, int] = {}
        self._point_components: dict[tuple[str, str], float] = {}

    def compile(self) -> DocumentSpec:
        source_text = self.clean_text
        source_line = 1
        if self.entry_macro:
            try:
                source_text, source_line = materialize_entry_macro(
                    self.clean_text,
                    self.entry_macro,
                )
            except MacroFrontendError as error:
                raise TikzNativeError(str(error)) from error
        pictures = [
            self._compile_picture(item)
            for item in self._extract_pictures(source_text, source_line=source_line)
        ]
        if self.entry_macro and not pictures:
            raise TikzNativeError(
                f"Entry macro \\{self.entry_macro} did not materialize a tikzpicture"
            )
        return DocumentSpec(
            source_path=(
                str(self.source_path)
                if self.source_path is not None
                else "<inline>"
            ),
            source_sha256=hashlib.sha256(self.source_text.encode("utf-8")).hexdigest(),
            colors=self.colors,
            pictures=pictures,
            entry_macro=self.entry_macro,
        )

    @staticmethod
    def _extract_colors(text: str) -> dict[str, str]:
        colors = {
            "black": "#000000",
            "white": "#FFFFFF",
            "gray": "#808080",
            "red": "#FF0000",
            "green": "#00FF00",
            "blue": "#0000FF",
            "cyan": "#00FFFF",
            "magenta": "#FF00FF",
            "yellow": "#FFFF00",
            "orange": "#FF8000",
            # xcolor's built-in ``purple`` is (0.75, 0, 0.25), not the
            # web/CSS purple often used by graphics libraries.
            "purple": "#BF0040",
        }
        pattern = re.compile(
            r"\\definecolor\{([^}]+)\}\{HTML\}\{([0-9A-Fa-f]{6})\}"
        )
        for name, value in pattern.findall(text):
            colors[name] = f"#{value.upper()}"
        return colors

    @staticmethod
    def _extract_zero_arg_gdefs(text: str) -> dict[str, str]:
        macros: dict[str, str] = {}
        pattern = re.compile(r"\\gdef\\([A-Za-z@]+)\s*\{")
        cursor = 0
        while match := pattern.search(text, cursor):
            body, end = _extract_balanced(text, match.end() - 1)
            macros[match.group(1)] = body
            cursor = end
        return macros

    def _extract_pictures(
        self,
        text: str | None = None,
        *,
        source_line: int = 1,
    ) -> list[_PictureSource]:
        picture_text = self.clean_text if text is None else text
        begin_re = re.compile(r"\\begin\{tikzpicture\}(?:\[([^\]]*)\])?")
        end_token = r"\end{tikzpicture}"
        pictures: list[_PictureSource] = []
        cursor = 0
        while match := begin_re.search(picture_text, cursor):
            end = picture_text.find(end_token, match.end())
            if end < 0:
                raise TikzNativeError("Unclosed tikzpicture environment")
            body = picture_text[match.end() : end]
            pictures.append(
                _PictureSource(
                    index=len(pictures) + 1,
                    start_line=source_line + _line_number(picture_text, match.start()) - 1,
                    end_line=source_line + _line_number(picture_text, end + len(end_token)) - 1,
                    options=match.group(1) or "",
                    body=body,
                    # A concrete figure wrapper often computes view-basis
                    # macros immediately before its tikzpicture.  Preserve
                    # only that local prefix; the unselected full macro
                    # library can contain unrelated parameter placeholders.
                    prelude=(
                        picture_text[cursor : match.start()]
                        if self.entry_macro
                        else ""
                    ),
                )
            )
            cursor = end + len(end_token)
        return pictures

    def _expand_code_macros(self, body: str) -> str:
        expanded = body
        for _ in range(8):
            changed = False
            for name, replacement in self.macros.items():
                pattern = re.compile(rf"\\{re.escape(name)}(?![A-Za-z@])")
                expanded, count = pattern.subn(lambda _match: replacement, expanded)
                changed = changed or bool(count)
            if not changed:
                break
        return self._expand_foreach(expanded)

    def _expand_foreach(self, text: str) -> str:
        pattern = re.compile(r"\\foreach\s+\\([A-Za-z]+)\s+in\s*\{")
        cursor = 0
        output: list[str] = []
        while match := pattern.search(text, cursor):
            output.append(text[cursor : match.start()])
            values_body, values_end = _extract_balanced(text, match.end() - 1)
            body_start = values_end
            while body_start < len(text) and text[body_start].isspace():
                body_start += 1
            if body_start >= len(text) or text[body_start] != "{":
                raise TikzNativeError("foreach body must be braced")
            loop_body, body_end = _extract_balanced(text, body_start)
            variable = match.group(1)
            for value in _split_top_level(values_body):
                output.append(
                    re.sub(
                        rf"\\{re.escape(variable)}(?![A-Za-z@])",
                        value.strip(),
                        loop_body,
                    )
                )
            cursor = body_end
        output.append(text[cursor:])
        return "".join(output)

    def _eval_expr(
        self,
        expression: str,
        symbols: dict[str, float | Length],
    ) -> float:
        value = expression.strip()
        value = value.replace("{", "(").replace("}", ")")
        value = value.replace("^", "**")

        def replace_point_component(match: re.Match[str]) -> str:
            key = (match.group(1).strip(), match.group(2))
            if key not in self._point_components:
                raise TikzNativeError(
                    f"Unknown TikZ point component {key[0]}.{key[1]} "
                    f"in {expression!r}"
                )
            return f"({self._point_components[key]:.17g})"

        value = re.sub(
            r"\\csname\s+pt3d@([^@\\\s]+)@([xyz])\\endcsname",
            replace_point_component,
            value,
        )

        def replace_symbol(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in symbols:
                raise TikzNativeError(f"Unknown PGF macro \\{name} in {expression!r}")
            symbol = symbols[name]
            if isinstance(symbol, Length):
                raise TikzNativeError(f"Length macro \\{name} used as a scalar")
            return f"({symbol:.17g})"

        value = re.sub(r"\\([A-Za-z@]+)", replace_symbol, value)
        value = re.sub(r"\s+", "", value)
        try:
            tree = ast.parse(value, mode="eval")
        except SyntaxError as error:
            raise TikzNativeError(f"Invalid PGF expression {expression!r}: {value!r}") from error
        return _SafeExpressionEvaluator().visit(tree)

    def _parse_length(
        self,
        value: str,
        symbols: dict[str, float | Length],
    ) -> Length:
        raw = value.strip()
        macro_match = re.fullmatch(r"\\([A-Za-z@]+)", raw)
        if macro_match:
            symbol = symbols.get(macro_match.group(1))
            if not isinstance(symbol, Length):
                raise TikzNativeError(f"Expected length macro in {raw!r}")
            return symbol
        unit_match = re.fullmatch(r"(.+?)(pt|mm|cm)", raw)
        if not unit_match:
            number = self._eval_expr(raw, symbols)
            return Length(number)
        number = self._eval_expr(unit_match.group(1), symbols)
        unit = unit_match.group(2)
        factor = {"pt": 1.0, "mm": MM_TO_PT, "cm": TEX_PT_PER_CM}[unit]
        return Length(number * factor)

    def _extract_math_macros(
        self,
        body: str,
        symbols: dict[str, float | Length],
    ) -> str:
        command_re = re.compile(r"\\pgfmathset(length)?macro\s*\{\\([A-Za-z@]+)\}\s*\{")
        cursor = 0
        output: list[str] = []
        while match := command_re.search(body, cursor):
            prefix = body[cursor : match.start()]
            output.append(prefix)
            self._record_direct_point_components(prefix, symbols)
            expression, end = _extract_balanced(body, match.end() - 1)
            name = match.group(2)
            if match.group(1):
                normalized = expression.strip()
                cm_match = re.fullmatch(r"(.+?)\*?1cm", normalized)
                if cm_match:
                    symbols[name] = Length(
                        self._eval_expr(cm_match.group(1), symbols) * TEX_PT_PER_CM
                    )
                else:
                    symbols[name] = self._parse_length(normalized, symbols)
            else:
                symbols[name] = self._eval_expr(expression, symbols)
            cursor = end
        suffix = body[cursor:]
        output.append(suffix)
        self._record_direct_point_components(suffix, symbols)
        return "".join(output)

    def _record_direct_point_components(
        self,
        text: str,
        symbols: dict[str, float | Length],
    ) -> None:
        """Remember concrete ``\\coordinate`` xyz values for PGF ``\\csname`` math.

        The handout's point helpers store components in TeX control sequences.
        Selected figures later read those components inside ``\\pgfmathsetmacro``.
        Recording the already-ordered direct declarations keeps that semantic
        dependency without attempting to execute arbitrary TeX.
        """

        coordinate_re = re.compile(
            r"\\coordinate\s*\(([^)]+)\)\s*at\s*\((.*?)\)\s*;",
            flags=re.DOTALL,
        )
        for match in coordinate_re.finditer(text):
            name = match.group(1).strip()
            parts = _split_top_level(match.group(2))
            if len(parts) != 3:
                continue
            try:
                values = tuple(self._eval_expr(part, symbols) for part in parts)
            except TikzNativeError:
                continue
            for axis, value in zip("xyz", values, strict=True):
                self._point_components[(name, axis)] = value

    def _resolve_ifdim_blocks(
        self,
        body: str,
        symbols: dict[str, float | Length],
    ) -> str:
        """Evaluate the simple PGF numeric conditionals used in figures.

        Text before each conditional is first interpreted so the condition can
        refer to a scalar assigned immediately above it.  This deliberately
        supports only ``<expr>pt <|> <expr>pt`` and balanced ``ifdim/else/fi``;
        arbitrary TeX execution remains outside the native compiler boundary.
        """

        output: list[str] = []
        remaining = body
        token_re = re.compile(r"\\ifdim|\\else|\\fi")
        while True:
            start = remaining.find(r"\ifdim")
            if start < 0:
                output.append(self._extract_math_macros(remaining, symbols))
                break
            output.append(self._extract_math_macros(remaining[:start], symbols))
            scoped = remaining[start:]
            depth = 0
            else_at: re.Match[str] | None = None
            end_at: re.Match[str] | None = None
            for token in token_re.finditer(scoped):
                if token.group(0) == r"\ifdim":
                    depth += 1
                elif token.group(0) == r"\else" and depth == 1:
                    else_at = token
                elif token.group(0) == r"\fi":
                    depth -= 1
                    if depth == 0:
                        end_at = token
                        break
            if end_at is None:
                raise TikzNativeError("unclosed \\ifdim block")

            condition_end = scoped.find("\n", len(r"\ifdim"))
            if condition_end < 0 or condition_end > end_at.start():
                raise TikzNativeError("\\ifdim condition must end at a newline")
            condition = scoped[len(r"\ifdim") : condition_end].strip()
            match = re.fullmatch(r"(.+?)pt\s*([<>])\s*(.+?)pt(?:\\relax)?", condition)
            if match is None:
                raise TikzNativeError(f"unsupported \\ifdim condition: {condition}")
            lhs = self._eval_expr(match.group(1), symbols)
            rhs = self._eval_expr(match.group(3), symbols)
            passes = lhs > rhs if match.group(2) == ">" else lhs < rhs
            branch_start = condition_end + 1
            true_end = (else_at or end_at).start()
            true_body = scoped[branch_start:true_end]
            false_body = (
                ""
                if else_at is None
                else scoped[else_at.end() : end_at.start()]
            )
            selected = true_body if passes else false_body
            # Selected content can itself contain a conditional.
            output.append(self._resolve_ifdim_blocks(selected, symbols))
            remaining = scoped[end_at.end() :]
        return "".join(output)

    @staticmethod
    def _replace_braced_command(
        text: str,
        command: str,
        group_count: int,
        replacement,
    ) -> str:
        pattern = re.compile(rf"\\{re.escape(command)}(?![A-Za-z@])")
        output: list[str] = []
        cursor = 0
        while match := pattern.search(text, cursor):
            groups: list[str] = []
            group_cursor = match.end()
            valid = True
            for _ in range(group_count):
                while group_cursor < len(text) and text[group_cursor].isspace():
                    group_cursor += 1
                if group_cursor >= len(text) or text[group_cursor] != "{":
                    valid = False
                    break
                group, group_cursor = _extract_balanced(text, group_cursor)
                groups.append(group.strip())
            if not valid:
                output.append(text[cursor : match.end()])
                cursor = match.end()
                continue
            output.append(text[cursor : match.start()])
            output.append(replacement(*groups))
            cursor = group_cursor
        output.append(text[cursor:])
        return "".join(output)

    def _normalize_space_point_commands(self, body: str) -> str:
        """Translate the handout's semantic 3D point helpers to TikZ coords."""

        normalized = self._replace_braced_command(
            body,
            "defPoint",
            4,
            lambda name, x, y, z: rf"\coordinate ({name}) at ({x},{y},{z});",
        )
        normalized = self._replace_braced_command(
            normalized,
            "defPointShift",
            5,
            lambda name, base, dx, dy, dz: (
                rf"\coordinate ({name}) at ($({base})+({dx},{dy},{dz})$);"
            ),
        )
        normalized = self._replace_braced_command(
            normalized,
            "pointOnSpaceLine",
            4,
            lambda name, start, end, parameter: (
                rf"\coordinate ({name}) at ($({start})!{parameter}!({end})$);"
            ),
        )
        return normalized

    def _picture_defaults(
        self,
        options: str,
        symbols: dict[str, float | Length],
    ) -> tuple[
        float,
        StyleSpec,
        Projection3DSpec | None,
        dict[str, str],
        list[str],
    ]:
        scale = 1.0
        style = StyleSpec()
        named_styles: dict[str, str] = {}
        warnings: list[str] = []
        basis: dict[str, Basis2] = {
            "x": (1.0, 0.0),
            "y": (0.0, 1.0),
            "z": (0.0, 0.0),
        }
        projection_source: str | None = None
        azimuth_degrees: float | None = None
        elevation_degrees: float | None = None
        for option in _split_top_level(options):
            key, _, value = option.partition("=")
            key = key.strip()
            value = value.strip()
            if key == "scale":
                scale = self._eval_expr(value, symbols)
            elif key == "space view":
                space_basis = self._parse_space_view(value, symbols)
                basis.update(zip(("x", "y", "z"), space_basis, strict=True))
                projection_source = "space view"
            elif key == "3d view":
                azimuth_degrees, elevation_degrees = self._parse_three_d_view(
                    value,
                    symbols,
                )
                view_basis = tikz_three_d_view_basis(
                    azimuth_degrees,
                    elevation_degrees,
                )
                basis.update(zip(("x", "y", "z"), view_basis, strict=True))
                projection_source = "3d view"
            elif key in {"x", "y", "z"}:
                basis[key] = self._parse_projection_basis(value, symbols)
                projection_source = "basis"
            elif key == "line width":
                style.line_width_pt = self._parse_length(value, symbols).pt
            elif key == "line cap":
                style.line_cap = value
            elif key == "line join":
                style.line_join = value
            elif key in {"ultra thin", "very thin", "thin", "semithick", "thick", "very thick", "ultra thick"}:
                style.line_width_pt = {
                    "ultra thin": 0.1,
                    "very thin": 0.2,
                    "thin": 0.4,
                    "semithick": 0.6,
                    "thick": 0.8,
                    "very thick": 1.2,
                    "ultra thick": 1.6,
                }[key]
            elif key.endswith("/.style"):
                style_name = key[: -len("/.style")].strip()
                if not style_name:
                    warnings.append(f"invalid empty picture style: {option}")
                else:
                    named_styles[style_name] = self._unbrace_option_value(value)
            elif key in {"baseline", "trim right", "trim left"}:
                warnings.append(f"layout-only picture option ignored: {option}")
            else:
                warnings.append(f"unhandled picture option: {option}")
        projection = None
        if projection_source is not None:
            projection = Projection3DSpec(
                source=projection_source,
                matrix=matrix_from_tikz_basis(
                    basis["x"],
                    basis["y"],
                    basis["z"],
                ),
                x_basis_cm=basis["x"],
                y_basis_cm=basis["y"],
                z_basis_cm=basis["z"],
                azimuth_degrees=azimuth_degrees,
                elevation_degrees=elevation_degrees,
            )
        return scale, style, projection, named_styles, warnings

    def _parse_space_view(
        self,
        value: str,
        symbols: dict[str, float | Length],
    ) -> tuple[Basis2, Basis2, Basis2]:
        raw = self._unbrace_option_value(value).strip()
        parts = _split_top_level(raw)
        if len(parts) != 3:
            raise TikzNativeError(
                f"space view requires x/y/z screen vectors: {value}"
            )
        vectors: list[Basis2] = []
        for part in parts:
            pair = part.strip()
            if not (pair.startswith("(") and pair.endswith(")")):
                raise TikzNativeError(f"space view vector must be parenthesized: {part}")
            components = _split_top_level(pair[1:-1])
            if len(components) != 2:
                raise TikzNativeError(f"space view vector requires two components: {part}")
            vectors.append(
                (
                    self._eval_expr(components[0], symbols),
                    self._eval_expr(components[1], symbols),
                )
            )
        return vectors[0], vectors[1], vectors[2]

    def _parse_projection_basis(
        self,
        value: str,
        symbols: dict[str, float | Length],
    ) -> Basis2:
        raw = self._unbrace_option_value(value).strip()
        if not (raw.startswith("(") and raw.endswith(")")):
            raise TikzNativeError(f"3D basis must be a coordinate pair: {value}")
        parts = _split_top_level(raw[1:-1])
        if len(parts) != 2:
            raise TikzNativeError(f"3D basis requires two screen components: {value}")
        return tuple(
            self._parse_length(part, symbols).pt / TEX_PT_PER_CM
            for part in parts
        )  # type: ignore[return-value]

    def _parse_three_d_view(
        self,
        value: str,
        symbols: dict[str, float | Length],
    ) -> tuple[float, float]:
        raw = value.strip()
        values: list[str] = []
        cursor = 0
        while cursor < len(raw):
            while cursor < len(raw) and raw[cursor].isspace():
                cursor += 1
            if cursor >= len(raw):
                break
            if raw[cursor] != "{":
                raise TikzNativeError(f"3d view expects two braced angles: {value}")
            item, cursor = _extract_balanced(raw, cursor)
            values.append(item)
        if len(values) != 2:
            raise TikzNativeError(f"3d view expects azimuth and elevation: {value}")
        return (
            self._eval_expr(values[0], symbols),
            self._eval_expr(values[1], symbols),
        )

    def _compile_picture(self, source: _PictureSource) -> PictureSpec:
        self._object_ids = {}
        self._point_components = {}
        symbols: dict[str, float | Length] = {}
        if source.prelude:
            self._extract_math_macros(source.prelude, symbols)
        normalized = self._normalize_space_point_commands(source.body)
        expanded = self._expand_code_macros(normalized)
        expanded = self._strip_pptstep_environments(expanded)
        expanded = self._normalize_canvas_scopes(expanded)
        expanded = self._terminate_semantic_commands(expanded)
        body = self._resolve_ifdim_blocks(expanded, symbols)
        scale, defaults, projection_3d, named_styles, warnings = self._picture_defaults(
            source.options,
            symbols,
        )
        picture = PictureSpec(
            index=source.index,
            start_line=source.start_line,
            end_line=source.end_line,
            raw_options=source.options,
            scale=scale,
            line_width_pt=defaults.line_width_pt,
            line_cap=defaults.line_cap,
            line_join=defaults.line_join,
            dimension=3 if projection_3d is not None else 2,
            projection_3d=projection_3d,
            named_styles=named_styles,
            symbols=symbols,
            warnings=warnings,
        )

        for scope_options in re.findall(
            r"\\begin\{scope\}\[([^\]]*)\]", body
        ):
            normalized = re.sub(r"\s+", "", scope_options)
            if normalized == "draw=none":
                picture.warnings.append(
                    "redundant scope option not propagated: draw=none"
                )
            else:
                picture.unsupported.append(
                    f"scope options require explicit native mapping: {scope_options}"
                )
        body = re.sub(r"\\begin\{scope\}(?:\[[^\]]*\])?", "", body)
        body = body.replace(r"\end{scope}", "")
        statements = self._split_statements(body)
        z_index = 0
        for statement in statements:
            statement = statement.strip()
            if not statement:
                continue
            z_index += 1
            try:
                objects = self._compile_statement(
                    statement,
                    picture,
                    defaults,
                    z_index,
                    source.start_line,
                )
                picture.objects.extend(objects)
            except TikzNativeError as error:
                picture.unsupported.append(f"{statement[:180]} :: {error}")
        return picture

    @staticmethod
    def _strip_pptstep_environments(body: str) -> str:
        """Animation metadata is retained by TeX but has no static geometry."""

        body = re.sub(r"\\begin\{pptstep\}(?:\[[^\]]*\])?", "", body)
        return body.replace(r"\end{pptstep}", "")

    def _normalize_canvas_scopes(self, body: str) -> str:
        """Lift the two plane-canvas node forms used by this document to xyz.

        TikZ interprets ``at (u,v)`` in these scopes as a coordinate in a
        selected world plane.  Lifting the coordinate preserves its anchor
        point, while the private ``native canvas plane`` option retains the
        local basis needed to apply ``transform shape`` in the renderer.
        """

        scope_re = re.compile(
            r"\\begin\{scope\}\[canvas is (yz|xy) plane at ([xz])=([^\]]+)\]"
            r"(.*?)\\end\{scope\}",
            flags=re.DOTALL,
        )

        def replace_scope(match: re.Match[str]) -> str:
            plane, fixed_axis, fixed_value, content = match.groups()
            fixed = fixed_value.strip()

            def lift_node(node_match: re.Match[str]) -> str:
                command, raw_options, spacing, first, second = node_match.groups()
                if plane == "yz" and fixed_axis == "x":
                    coordinate = f"({fixed},{first},{second})"
                elif plane == "xy" and fixed_axis == "z":
                    coordinate = f"({first},{second},{fixed})"
                else:
                    return node_match.group(0)
                if raw_options:
                    options = (
                        raw_options[:-1]
                        + f",native canvas plane={plane}]"
                    )
                else:
                    options = f"[native canvas plane={plane}]"
                return command + options + spacing + coordinate

            return re.sub(
                r"(\\node)(\[[^\]]*\])?(\s+at\s*)"
                r"\(([^,()]+),([^,()]+)\)",
                lift_node,
                content,
            )

        previous = None
        while previous != body:
            previous = body
            body = scope_re.sub(replace_scope, body)
        return body

    @staticmethod
    def _terminate_semantic_commands(body: str) -> str:
        """Give semicolon-free handout geometry helpers statement boundaries."""

        signatures = {
            "DrawSpaceLineBehindHorizontalFace": (2, 6),
            "DrawSpaceLineBehindTriFace": (2, 5),
            "DrawSpaceLineBehindParallelogramFace": (2, 6),
            "DrawSpacePlaneInteraction": (6, 2),
            "DeclareSpaceHinge": (0, 4),
            "setSpaceOcclusionProjection": (0, 6),
        }
        for name, (optional_count, group_count) in signatures.items():
            pattern = re.compile(rf"\\{name}(?![A-Za-z@])")
            cursor = 0
            output: list[str] = []
            while match := pattern.search(body, cursor):
                output.append(body[cursor : match.start()])
                end = match.end()
                for _ in range(optional_count):
                    scan = end
                    while scan < len(body) and body[scan].isspace():
                        scan += 1
                    if scan < len(body) and body[scan] == "[":
                        _value, end = _extract_balanced(body, scan, "[", "]")
                valid = True
                for _ in range(group_count):
                    while end < len(body) and body[end].isspace():
                        end += 1
                    if end >= len(body) or body[end] != "{":
                        valid = False
                        break
                    _value, end = _extract_balanced(body, end)
                if not valid:
                    output.append(body[match.start() : match.end()])
                    cursor = match.end()
                    continue
                output.append(body[match.start() : end])
                if not body[end:].lstrip().startswith(";"):
                    output.append(";")
                cursor = end
            output.append(body[cursor:])
            body = "".join(output)
        return body

    @staticmethod
    def _split_statements(body: str) -> list[str]:
        statements: list[str] = []
        buffer: list[str] = []
        stack: list[str] = []
        pairs = {"{": "}", "[": "]", "(": ")"}
        closers = set(pairs.values())
        escaped = False
        for char in body:
            if escaped:
                buffer.append(char)
                escaped = False
                continue
            if char == "\\":
                buffer.append(char)
                escaped = True
                continue
            if char in pairs:
                stack.append(pairs[char])
            elif char in closers and stack and char == stack[-1]:
                stack.pop()
            if char == ";" and not stack:
                statements.append("".join(buffer))
                buffer = []
            else:
                buffer.append(char)
        tail = "".join(buffer).strip()
        if tail:
            statements.append(tail)
        return statements

    def _compile_statement(
        self,
        statement: str,
        picture: PictureSpec,
        defaults: StyleSpec,
        z_index: int,
        source_line: int,
    ) -> list[ObjectSpec]:
        statement = statement.strip()
        command_match = re.match(
            r"\\(coordinate|path|draw|filldraw|fill|node|pic|"
            r"DrawSpaceLineBehindHorizontalFace|DrawSpaceLineBehindTriFace|"
            r"DrawSpaceLineBehindParallelogramFace|DrawSpacePlaneInteraction|"
            r"DeclareSpaceHinge|"
            r"setSpaceOcclusionProjection)\b",
            statement,
        )
        if not command_match:
            if statement:
                raise TikzNativeError("statement contains no supported TikZ command")
            return []
        command = command_match.group(1)
        if command == "DeclareSpaceHinge":
            self._compile_space_hinge_declaration(
                statement,
                picture,
                source_line,
            )
            return []
        if command.startswith("DrawSpace"):
            return self._compile_space_semantic_command(
                command,
                statement,
                picture,
                defaults,
                z_index,
                source_line,
            )
        if command == "setSpaceOcclusionProjection":
            return []
        if command == "coordinate":
            self._compile_coordinate(statement, picture)
            return []
        if command == "path":
            options, _rest = self._extract_options(statement, "path")
            if "node" in statement and not re.search(
                r"\bname\s+(?:path|intersections)\s*=",
                options,
            ):
                # A plain \path can carry labels without drawing its carrier.
                # Parse it through the native path machinery, then retain only
                # the semantic label objects.
                candidates = self._compile_path_command(
                    "draw",
                    r"\draw" + statement[len(r"\path") :],
                    picture,
                    defaults,
                    z_index,
                    source_line,
                )
                return [
                    item
                    for item in candidates
                    if item.kind in {"label", "path_label"}
                ]
            self._compile_construction_path(statement, picture, source_line)
            return []
        if command == "node":
            return [self._compile_standalone_node(statement, picture, defaults, z_index, source_line)]
        if command == "pic":
            return self._compile_pic(statement, picture, defaults, z_index, source_line)
        return self._compile_path_command(
            command,
            statement,
            picture,
            defaults,
            z_index,
            source_line,
        )

    def _compile_coordinate(self, statement: str, picture: PictureSpec) -> None:
        match = re.fullmatch(
            r"\\coordinate\s*\(([^)]+)\)\s*at\s*(.+)",
            statement.strip(),
            flags=re.DOTALL,
        )
        if not match:
            raise TikzNativeError("unsupported coordinate declaration")
        name = match.group(1).strip()
        coordinate = self._parse_coord(match.group(2).strip(), picture)
        picture.coordinates[name] = coordinate.xy
        if coordinate.dependency is not None:
            picture.coordinate_dependencies[name] = coordinate.dependency

    @staticmethod
    def _parse_semantic_arguments(
        statement: str,
        command: str,
        optional_defaults: list[str],
        required_count: int,
    ) -> tuple[list[str], list[str]]:
        prefix = rf"\{command}"
        if not statement.startswith(prefix):
            raise TikzNativeError(f"expected {prefix}")
        cursor = len(prefix)
        optional: list[str] = []
        for default in optional_defaults:
            while cursor < len(statement) and statement[cursor].isspace():
                cursor += 1
            if cursor < len(statement) and statement[cursor] == "[":
                value, cursor = _extract_balanced(statement, cursor, "[", "]")
                optional.append(value.strip())
            else:
                optional.append(default)
        required: list[str] = []
        for _ in range(required_count):
            while cursor < len(statement) and statement[cursor].isspace():
                cursor += 1
            if cursor >= len(statement) or statement[cursor] != "{":
                raise TikzNativeError(f"missing argument to {prefix}")
            value, cursor = _extract_balanced(statement, cursor)
            required.append(value.strip())
        if statement[cursor:].strip():
            raise TikzNativeError(f"unexpected tokens after {prefix}")
        return optional, required

    def _compile_space_hinge_declaration(
        self,
        statement: str,
        picture: PictureSpec,
        source_line: int,
    ) -> None:
        """Record ``\\DeclareSpaceHinge`` without emitting visible geometry."""

        _optional, required = self._parse_semantic_arguments(
            statement,
            "DeclareSpaceHinge",
            [],
            4,
        )
        relation_id = required[0].strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}", relation_id):
            raise TikzNativeError(
                "DeclareSpaceHinge relation ID must be a portable identifier"
            )
        if any(item.id == relation_id for item in picture.hinge_relations):
            raise TikzNativeError(
                f"duplicate DeclareSpaceHinge relation ID: {relation_id}"
            )

        def names(raw: str, field: str, minimum: int) -> list[str]:
            values = [item.strip() for item in raw.split("/") if item.strip()]
            if len(values) < minimum or len(values) != len(set(values)):
                raise TikzNativeError(
                    f"DeclareSpaceHinge {field} must contain at least {minimum} "
                    "distinct named coordinates"
                )
            missing = [name for name in values if name not in picture.coordinates]
            if missing:
                raise TikzNativeError(
                    f"DeclareSpaceHinge {field} uses unknown coordinates: "
                    + ", ".join(missing)
                )
            if any(len(picture.coordinates[name]) != 3 for name in values):
                raise TikzNativeError(
                    f"DeclareSpaceHinge {field} requires three-dimensional coordinates"
                )
            return values

        axis = names(required[1], "axis", 2)
        if len(axis) != 2:
            raise TikzNativeError(
                "DeclareSpaceHinge axis must contain exactly two named coordinates"
            )
        fixed_face = names(required[2], "fixed face", 3)
        moving_face = names(required[3], "moving face", 3)
        if not set(axis).issubset(fixed_face) or not set(axis).issubset(moving_face):
            raise TikzNativeError(
                "DeclareSpaceHinge axis endpoints must belong to both faces"
            )
        if not (set(fixed_face) - set(axis)):
            raise TikzNativeError(
                "DeclareSpaceHinge fixed face needs a point outside the axis"
            )
        if not (set(moving_face) - set(axis)):
            raise TikzNativeError(
                "DeclareSpaceHinge moving face needs a point outside the axis"
            )
        axis_start = picture.coordinates[axis[0]]
        axis_end = picture.coordinates[axis[1]]
        if sum((axis_end[index] - axis_start[index]) ** 2 for index in range(3)) <= 1e-18:
            raise TikzNativeError("DeclareSpaceHinge axis must not have zero length")

        axis_vector = tuple(
            axis_end[index] - axis_start[index] for index in range(3)
        )
        axis_norm_squared = sum(component * component for component in axis_vector)

        def face_has_off_axis_point(face: list[str]) -> bool:
            for name in face:
                if name in axis:
                    continue
                relative = tuple(
                    picture.coordinates[name][index] - axis_start[index]
                    for index in range(3)
                )
                projection = sum(
                    relative[index] * axis_vector[index] for index in range(3)
                ) / axis_norm_squared
                perpendicular = tuple(
                    relative[index] - projection * axis_vector[index]
                    for index in range(3)
                )
                if sum(component * component for component in perpendicular) > 1e-18:
                    return True
            return False

        if not face_has_off_axis_point(fixed_face):
            raise TikzNativeError(
                "DeclareSpaceHinge fixed face needs a point outside the axis"
            )
        if not face_has_off_axis_point(moving_face):
            raise TikzNativeError(
                "DeclareSpaceHinge moving face needs a point outside the axis"
            )
        picture.hinge_relations.append(
            HingeRelationSpec(
                id=relation_id,
                axis_names=axis,
                fixed_face_names=fixed_face,
                moving_face_names=moving_face,
                source_line=source_line,
                raw=re.sub(r"\s+", " ", statement).strip(),
            )
        )

    def _occlusion_interval(
        self,
        start: tuple[float, ...],
        end: tuple[float, ...],
        face: list[tuple[float, ...]],
        picture: PictureSpec,
    ) -> tuple[float, float] | None:
        """Return the part of a segment hidden by a triangle/parallelogram."""

        if picture.projection_3d is None or len(start) != 3 or len(end) != 3:
            return None
        if len(face) not in {3, 4} or any(len(point) != 3 for point in face):
            return None
        try:
            return parallel_occlusion_interval(
                start,
                end,
                face,
                picture.projection_3d.matrix[2],
            )
        except OcclusionGeometryError as error:
            raise TikzNativeError(
                f"invalid 3D occlusion relation: {error}"
            ) from error

    @staticmethod
    def _interpolate_point(
        start: tuple[float, ...],
        end: tuple[float, ...],
        parameter: float,
    ) -> tuple[float, ...]:
        return tuple(
            start[index] + parameter * (end[index] - start[index])
            for index in range(len(start))
        )

    def _compile_occluded_line(
        self,
        *,
        start_name: str,
        end_name: str,
        face_names: list[str],
        visible_options: str,
        hidden_options: str,
        picture: PictureSpec,
        defaults: StyleSpec,
        z_index: int,
        source_line: int,
        raw: str,
    ) -> list[ObjectSpec]:
        start = self._lookup_named(start_name, picture)
        end = self._lookup_named(end_name, picture)
        face = [self._lookup_named(name, picture) for name in face_names]
        visible_style = self._parse_style(
            visible_options,
            "draw",
            defaults,
            picture,
        )
        hidden_style = self._parse_style(
            hidden_options,
            "draw",
            defaults,
            picture,
        )
        interval = self._occlusion_interval(start, end, face, picture)
        relation_id = self._semantic_id(
            "occlusion_relation",
            [start_name, end_name, *face_names],
        )
        if interval is None:
            segments = [(0.0, 1.0, visible_style, "visible")]
        else:
            start_hidden, end_hidden = interval
            segments: list[tuple[float, float, StyleSpec, str]] = []
            if start_hidden > 1e-7:
                segments.append((0.0, start_hidden, visible_style, "visible"))
            segments.append((start_hidden, end_hidden, hidden_style, "hidden"))
            if end_hidden < 1.0 - 1e-7:
                segments.append((end_hidden, 1.0, visible_style, "visible"))
        objects: list[ObjectSpec] = []
        for index, (first_t, last_t, style, visibility) in enumerate(segments):
            objects.append(
                self._object(
                    self._semantic_id(
                        f"occluded_{visibility}",
                        [start_name, end_name, str(index)],
                    ),
                    "line",
                    {
                        "start": self._interpolate_point(start, end, first_t),
                        "end": self._interpolate_point(start, end, last_t),
                        "start_name": start_name,
                        "end_name": end_name,
                        "source_parameter_range": (first_t, last_t),
                        "occluding_face": face_names,
                        "visibility": visibility,
                    },
                    style,
                    z_index,
                    source_line,
                    raw,
                )
            )
        picture.occlusion_relations.append(
            OcclusionRelationSpec(
                id=relation_id,
                start_name=start_name,
                end_name=end_name,
                face_names=list(face_names),
                visible_style=visible_style,
                hidden_style=hidden_style,
                object_ids=[item.id for item in objects],
                z_index=z_index,
                source_line=source_line,
                raw=re.sub(r"\s+", " ", raw).strip(),
            )
        )
        return objects

    def _compile_space_semantic_command(
        self,
        command: str,
        statement: str,
        picture: PictureSpec,
        defaults: StyleSpec,
        z_index: int,
        source_line: int,
    ) -> list[ObjectSpec]:
        if command in {
            "DrawSpaceLineBehindHorizontalFace",
            "DrawSpaceLineBehindParallelogramFace",
        }:
            optional, required = self._parse_semantic_arguments(
                statement,
                command,
                ["edge", "hidden"],
                6,
            )
            return self._compile_occluded_line(
                start_name=required[0],
                end_name=required[1],
                face_names=required[2:],
                visible_options=optional[0],
                hidden_options=optional[1],
                picture=picture,
                defaults=defaults,
                z_index=z_index,
                source_line=source_line,
                raw=statement,
            )
        if command == "DrawSpaceLineBehindTriFace":
            optional, required = self._parse_semantic_arguments(
                statement,
                command,
                ["edge", "hidden"],
                5,
            )
            return self._compile_occluded_line(
                start_name=required[0],
                end_name=required[1],
                face_names=required[2:],
                visible_options=optional[0],
                hidden_options=optional[1],
                picture=picture,
                defaults=defaults,
                z_index=z_index,
                source_line=source_line,
                raw=statement,
            )

        optional, required = self._parse_semantic_arguments(
            statement,
            command,
            [
                "planePrime",
                "planePrimeCovered",
                "planeborder",
                "planeHidden",
                "betaEdge",
                "betaEdgeHidden",
            ],
            2,
        )
        alpha_names = [item.strip() for item in required[0].split("/")]
        beta_names = [item.strip() for item in required[1].split("/")]
        if len(alpha_names) != 4 or len(beta_names) != 4:
            raise TikzNativeError("plane interaction requires two four-point faces")
        beta_points = [self._lookup_named(name, picture) for name in beta_names]
        beta_style = self._parse_style(optional[0], "fill", defaults, picture)
        objects = [
            self._object(
                self._semantic_id("plane_interaction_fill", beta_names),
                "polygon",
                {
                    "points": beta_points,
                    "point_names": list(beta_names),
                    "closed": True,
                    "semantic": "beta_plane",
                },
                beta_style,
                z_index,
                source_line,
                statement,
            )
        ]
        for index in range(4):
            objects.extend(
                self._compile_occluded_line(
                    start_name=beta_names[index],
                    end_name=beta_names[(index + 1) % 4],
                    face_names=alpha_names,
                    visible_options=optional[4],
                    hidden_options=optional[5],
                    picture=picture,
                    defaults=defaults,
                    z_index=z_index + 1,
                    source_line=source_line,
                    raw=statement,
                )
            )
        for index in range(4):
            objects.extend(
                self._compile_occluded_line(
                    start_name=alpha_names[index],
                    end_name=alpha_names[(index + 1) % 4],
                    face_names=beta_names,
                    visible_options=optional[2],
                    hidden_options=optional[3],
                    picture=picture,
                    defaults=defaults,
                    z_index=z_index + 2,
                    source_line=source_line,
                    raw=statement,
                )
            )
        return objects

    @staticmethod
    def _unbrace_option_value(value: str) -> str:
        text = value.strip()
        if not text.startswith("{"):
            return text
        body, end = _extract_balanced(text, 0)
        if text[end:].strip():
            raise TikzNativeError(f"unexpected tokens after braced option: {value!r}")
        return body

    def _compile_construction_path(
        self,
        statement: str,
        picture: PictureSpec,
        source_line: int,
    ) -> None:
        options, path = self._extract_options(statement, "path")
        parsed_options: dict[str, str] = {}
        for option in _split_top_level(options):
            key, separator, value = option.partition("=")
            if not separator:
                raise TikzNativeError(f"construction path option requires a value: {option}")
            parsed_options[key.strip()] = value.strip()

        path_name = parsed_options.get("name path")
        intersection_options = parsed_options.get("name intersections")
        if path_name and intersection_options:
            raise TikzNativeError("a path cannot declare a name and intersections together")
        if path_name:
            name = self._unbrace_option_value(path_name).strip()
            if not name:
                raise TikzNativeError("named path requires a nonempty name")
            if name in picture.named_paths:
                raise TikzNativeError(f"duplicate named path: {name}")
            kind, geometry = self._parse_named_path_geometry(path, picture)
            picture.named_paths[name] = NamedPathSpec(
                name=name,
                kind=kind,
                geometry=geometry,
                source_line=source_line,
                raw=statement,
            )
            return
        if intersection_options:
            if path.strip():
                raise TikzNativeError(
                    "intersection construction with trailing path operations is not supported"
                )
            self._compile_named_intersections(
                self._unbrace_option_value(intersection_options),
                statement,
                picture,
                source_line,
            )
            return
        raise TikzNativeError(
            "construction path requires 'name path' or 'name intersections'"
        )

    def _parse_named_path_geometry(
        self,
        path: str,
        picture: PictureSpec,
    ) -> tuple[str, dict[str, Any]]:
        ellipse_match = re.fullmatch(
            r"(.+?)\s+ellipse\s*\[([^\]]+)\]\s*",
            path,
            flags=re.DOTALL,
        )
        if ellipse_match:
            center = self._parse_coord(ellipse_match.group(1).strip(), picture)
            radii = dict(
                part.split("=", 1)
                for part in _split_top_level(ellipse_match.group(2))
            )
            try:
                rx = self._eval_expr(radii["x radius"], picture.symbols)
                ry = self._eval_expr(radii["y radius"], picture.symbols)
            except KeyError as error:
                raise TikzNativeError(
                    "named ellipse requires x radius and y radius"
                ) from error
            if rx <= 0 or ry <= 0:
                raise TikzNativeError("named ellipse radii must be positive")
            return (
                "ellipse",
                {
                    "center": center.xy,
                    "center_name": center.name,
                    "rx": rx,
                    "ry": ry,
                },
            )

        circle_match = re.fullmatch(
            r"(.+?)\s+circle\s*\(([^)]+)\)\s*",
            path,
            flags=re.DOTALL,
        )
        if circle_match:
            center = self._parse_coord(circle_match.group(1).strip(), picture)
            radius = self._eval_expr(circle_match.group(2), picture.symbols)
            if radius <= 0:
                raise TikzNativeError("named circle radius must be positive")
            return (
                "ellipse",
                {
                    "center": center.xy,
                    "center_name": center.name,
                    "rx": radius,
                    "ry": radius,
                },
            )

        path_without_nodes, node_specs = self._extract_inline_nodes(path)
        if node_specs:
            raise TikzNativeError("nodes on named construction paths are not supported")
        coordinates = self._find_path_coordinates(path_without_nodes, picture)
        if len(coordinates) != 2 or not re.search(r"--", path_without_nodes):
            raise TikzNativeError(
                "named construction paths currently support one line segment, circle, or ellipse"
            )
        start, end = coordinates[0][0], coordinates[1][0]
        if start.xy == end.xy:
            raise TikzNativeError("named line path cannot have zero length")
        return (
            "line",
            {
                "start": start.xy,
                "end": end.xy,
                "start_name": start.name,
                "end_name": end.name,
                "start_dependency": start.dependency,
                "end_dependency": end.dependency,
            },
        )

    def _compile_named_intersections(
        self,
        options: str,
        statement: str,
        picture: PictureSpec,
        source_line: int,
    ) -> None:
        if picture.dimension == 3:
            raise TikzNativeError(
                "3D named-path intersections require an explicit spatial relation"
            )
        parsed: dict[str, str] = {}
        for option in _split_top_level(options):
            key, separator, value = option.partition("=")
            if not separator:
                raise TikzNativeError(
                    f"name intersections option requires a value: {option}"
                )
            parsed[key.strip()] = value.strip()

        if "of" not in parsed or "by" not in parsed:
            raise TikzNativeError("name intersections requires both 'of' and 'by'")
        path_names = re.split(r"\s+and\s+", parsed["of"].strip(), maxsplit=1)
        if len(path_names) != 2:
            raise TikzNativeError("intersection 'of' must be 'path A and path B'")
        path_a, path_b = (name.strip() for name in path_names)
        if path_a not in picture.named_paths or path_b not in picture.named_paths:
            missing = [
                name
                for name in (path_a, path_b)
                if name not in picture.named_paths
            ]
            raise TikzNativeError(
                "unknown named path in intersection: " + ", ".join(missing)
            )

        sort_by = parsed.get("sort by", path_a).strip()
        if sort_by not in {path_a, path_b}:
            raise TikzNativeError(
                f"sort by must name one intersected path, got {sort_by!r}"
            )
        if picture.named_paths[sort_by].kind != "line":
            raise TikzNativeError(
                "deterministic native conversion currently requires sorting intersections "
                "by an oriented line path"
            )

        names_body = self._unbrace_option_value(parsed["by"])
        coordinate_names = [
            item.strip() for item in _split_top_level(names_body) if item.strip()
        ]
        if not coordinate_names:
            raise TikzNativeError("intersection 'by' list cannot be empty")

        solutions = self._intersect_named_paths(
            path_a,
            picture.named_paths[path_a],
            path_b,
            picture.named_paths[path_b],
        )
        solutions.sort(key=lambda solution: solution[1][sort_by])
        if len(coordinate_names) > len(solutions):
            raise TikzNativeError(
                f"requested {len(coordinate_names)} named intersections but found "
                f"{len(solutions)}"
            )

        assigned = solutions[: len(coordinate_names)]
        points: list[tuple[float, float]] = []
        sort_parameters: list[float] = []
        for index, (name, solution) in enumerate(zip(coordinate_names, assigned)):
            point, parameters = solution
            picture.coordinates[name] = point
            picture.coordinate_dependencies[name] = {
                "operation": "intersection",
                "path_a": path_a,
                "path_b": path_b,
                "sort_by": sort_by,
                "sorted_index": index,
                "sort_parameter": parameters[sort_by],
            }
            points.append(point)
            sort_parameters.append(parameters[sort_by])

        picture.intersections.append(
            IntersectionSpec(
                path_a=path_a,
                path_b=path_b,
                sort_by=sort_by,
                coordinate_names=coordinate_names,
                points=points,
                sort_parameters=sort_parameters,
                source_line=source_line,
                raw=statement,
            )
        )

    def _intersect_named_paths(
        self,
        name_a: str,
        path_a: NamedPathSpec,
        name_b: str,
        path_b: NamedPathSpec,
    ) -> list[tuple[tuple[float, float], dict[str, float]]]:
        if path_a.kind == "line" and path_b.kind == "ellipse":
            return self._intersect_line_ellipse(name_a, path_a, name_b, path_b)
        if path_a.kind == "ellipse" and path_b.kind == "line":
            return self._intersect_line_ellipse(name_b, path_b, name_a, path_a)
        if path_a.kind == "line" and path_b.kind == "line":
            return self._intersect_lines(name_a, path_a, name_b, path_b)
        raise TikzNativeError(
            f"unsupported named-path intersection: {path_a.kind} with {path_b.kind}"
        )

    @staticmethod
    def _intersect_line_ellipse(
        line_name: str,
        line: NamedPathSpec,
        ellipse_name: str,
        ellipse: NamedPathSpec,
    ) -> list[tuple[tuple[float, float], dict[str, float]]]:
        x0, y0 = line.geometry["start"]
        x1, y1 = line.geometry["end"]
        cx, cy = ellipse.geometry["center"]
        rx = ellipse.geometry["rx"]
        ry = ellipse.geometry["ry"]
        dx, dy = x1 - x0, y1 - y0
        ox, oy = x0 - cx, y0 - cy
        quadratic_a = dx * dx / (rx * rx) + dy * dy / (ry * ry)
        quadratic_b = 2 * (ox * dx / (rx * rx) + oy * dy / (ry * ry))
        quadratic_c = ox * ox / (rx * rx) + oy * oy / (ry * ry) - 1
        discriminant = quadratic_b * quadratic_b - 4 * quadratic_a * quadratic_c
        tolerance = 1e-12
        if discriminant < -tolerance:
            return []
        root = math.sqrt(max(discriminant, 0.0))
        parameters = [(-quadratic_b - root) / (2 * quadratic_a)]
        if root > tolerance:
            parameters.append((-quadratic_b + root) / (2 * quadratic_a))
        solutions = []
        for line_parameter in parameters:
            point = (
                x0 + line_parameter * dx,
                y0 + line_parameter * dy,
            )
            ellipse_parameter = math.atan2(
                (point[1] - cy) / ry,
                (point[0] - cx) / rx,
            ) % (2 * math.pi)
            solutions.append(
                (
                    point,
                    {
                        line_name: line_parameter,
                        ellipse_name: ellipse_parameter,
                    },
                )
            )
        return solutions

    @staticmethod
    def _intersect_lines(
        name_a: str,
        path_a: NamedPathSpec,
        name_b: str,
        path_b: NamedPathSpec,
    ) -> list[tuple[tuple[float, float], dict[str, float]]]:
        ax0, ay0 = path_a.geometry["start"]
        ax1, ay1 = path_a.geometry["end"]
        bx0, by0 = path_b.geometry["start"]
        bx1, by1 = path_b.geometry["end"]
        adx, ady = ax1 - ax0, ay1 - ay0
        bdx, bdy = bx1 - bx0, by1 - by0
        denominator = adx * bdy - ady * bdx
        if abs(denominator) <= 1e-12:
            return []
        offset_x, offset_y = bx0 - ax0, by0 - ay0
        parameter_a = (offset_x * bdy - offset_y * bdx) / denominator
        parameter_b = (offset_x * ady - offset_y * adx) / denominator
        point = (ax0 + parameter_a * adx, ay0 + parameter_a * ady)
        return [(point, {name_a: parameter_a, name_b: parameter_b})]

    def _parse_coord(self, raw: str, picture: PictureSpec) -> _CoordValue:
        text = raw.strip()
        if text.startswith("$(") and text.endswith(")$"):
            text = text[1:-1]
        if text.startswith("(") and text.endswith(")"):
            inner = text[1:-1].strip()
        else:
            raise TikzNativeError(f"coordinate must be parenthesized: {raw!r}")

        if inner.startswith("$") and inner.endswith("$"):
            inner = inner[1:-1].strip()

        if inner in picture.coordinates:
            return _CoordValue(
                picture.coordinates[inner],
                inner,
                {"operation": "reference", "coordinate": inner},
            )

        projection = re.fullmatch(r"\(([^)]+)\)!\(([^)]+)\)!\(([^)]+)\)", inner)
        if projection:
            start = self._lookup_named(projection.group(1), picture)
            point = self._lookup_named(projection.group(2), picture)
            end = self._lookup_named(projection.group(3), picture)
            if not (len(start) == len(point) == len(end)):
                raise TikzNativeError("projection coordinates use mixed dimensions")
            direction = tuple(
                end[index] - start[index] for index in range(len(start))
            )
            denominator = sum(value * value for value in direction)
            if denominator == 0:
                raise TikzNativeError("cannot project onto a zero-length line")
            t = sum(
                (point[index] - start[index]) * direction[index]
                for index in range(len(start))
            ) / denominator
            return _CoordValue(
                tuple(
                    start[index] + t * direction[index]
                    for index in range(len(start))
                ),
                dependency={
                    "operation": "projection",
                    "line_start": projection.group(1).strip(),
                    "point": projection.group(2).strip(),
                    "line_end": projection.group(3).strip(),
                },
            )

        interpolation = re.fullmatch(r"\(([^)]+)\)!([^!]+)!\(([^)]+)\)", inner)
        if interpolation:
            start = self._lookup_named(interpolation.group(1), picture)
            end = self._lookup_named(interpolation.group(3), picture)
            if len(start) != len(end):
                raise TikzNativeError("interpolation coordinates use mixed dimensions")
            t = self._eval_expr(interpolation.group(2), picture.symbols)
            return _CoordValue(
                tuple(
                    start[index] + t * (end[index] - start[index])
                    for index in range(len(start))
                ),
                dependency={
                    "operation": "interpolation",
                    "start": interpolation.group(1).strip(),
                    "end": interpolation.group(3).strip(),
                    "parameter": t,
                    "parameter_expression": interpolation.group(2).strip(),
                },
            )

        addition = re.fullmatch(r"\(([^)]+)\)\+\((.+)\)", inner)
        if addition:
            base = self._lookup_named(addition.group(1), picture)
            parts = _split_top_level(addition.group(2))
            if len(parts) == 2 and len(base) == 3:
                parts.append("0")
            if len(parts) != len(base):
                raise TikzNativeError(
                    f"coordinate addition expects {len(base)} components"
                )
            offset = tuple(
                self._eval_expr(part, picture.symbols) for part in parts
            )
            return _CoordValue(
                tuple(
                    base[index] + offset[index] for index in range(len(base))
                ),
                dependency={
                    "operation": "translation",
                    "base": addition.group(1).strip(),
                    "offset": offset,
                },
            )

        parts = _split_top_level(inner)
        if len(parts) in {2, 3}:
            values = tuple(
                self._eval_expr(part, picture.symbols) for part in parts
            )
            if len(values) == 3:
                picture.dimension = 3
                if picture.projection_3d is None:
                    raise TikzNativeError(
                        "3D coordinate requires '3d view' or explicit x/y/z bases"
                    )
            elif picture.dimension == 3:
                values = (*values, 0.0)
            return _CoordValue(values)
        raise TikzNativeError(f"unsupported coordinate expression: {raw!r}")

    @staticmethod
    def _lookup_named(name: str, picture: PictureSpec) -> tuple[float, ...]:
        clean = name.strip()
        if clean not in picture.coordinates:
            raise TikzNativeError(f"unknown named coordinate: {clean}")
        return picture.coordinates[clean]

    def _extract_options(self, text: str, command: str) -> tuple[str, str]:
        prefix = f"\\{command}"
        if not text.startswith(prefix):
            raise TikzNativeError(f"expected {prefix}")
        rest = text[len(prefix) :].lstrip()
        if rest.startswith("["):
            options, end = _extract_balanced(rest, 0, "[", "]")
            return options, rest[end:].strip()
        return "", rest

    def _parse_style(
        self,
        options: str,
        command: str,
        defaults: StyleSpec,
        picture: PictureSpec,
    ) -> StyleSpec:
        style = StyleSpec(
            draw_color=None if command == "fill" else defaults.draw_color,
            fill_color=None if command == "node" else defaults.fill_color,
            opacity=defaults.opacity,
            line_width_pt=defaults.line_width_pt,
            line_cap=defaults.line_cap,
            line_join=defaults.line_join,
            inner_xsep_pt=defaults.inner_xsep_pt,
            inner_ysep_pt=defaults.inner_ysep_pt,
            transform_shape=defaults.transform_shape,
            native_canvas_plane=defaults.native_canvas_plane,
            rectangle_node=defaults.rectangle_node,
            rotate_degrees=defaults.rotate_degrees,
        )
        for option in self._expand_named_style_options(options, picture):
            style.raw_options.append(option)
            key, separator, value = option.partition("=")
            key = key.strip()
            value = value.strip()
            if key == "line width" and separator:
                style.line_width_pt = self._parse_length(value, picture.symbols).pt
            elif key in {
                "ultra thin",
                "very thin",
                "thin",
                "semithick",
                "thick",
                "very thick",
                "ultra thick",
            }:
                style.line_width_pt = {
                    "ultra thin": 0.1,
                    "very thin": 0.2,
                    "thin": 0.4,
                    "semithick": 0.6,
                    "thick": 0.8,
                    "very thick": 1.2,
                    "ultra thick": 1.6,
                }[key]
            elif key == "opacity" and separator:
                style.opacity = self._eval_expr(value, picture.symbols)
            elif key == "fill opacity" and separator:
                style.fill_opacity = self._eval_expr(value, picture.symbols)
            elif key == "draw opacity" and separator:
                style.draw_opacity = self._eval_expr(value, picture.symbols)
            elif key == "draw" and separator:
                style.draw_color = None if value == "none" else self._resolve_color(value)
            elif key == "draw" and not separator:
                # Bare ``draw`` is TikZ's request to use the current/default
                # drawing colour; the native style already carries it.
                pass
            elif key == "fill" and separator:
                style.fill_color = None if value == "none" else self._resolve_color(value)
            elif key == "font" and separator:
                style.font_command = value
            elif key == "transform shape":
                style.transform_shape = True
            elif key == "native canvas plane" and separator:
                if value not in {"xy", "yz"}:
                    raise TikzNativeError(f"unsupported native canvas plane: {value}")
                style.native_canvas_plane = value
            elif key == "rectangle":
                style.rectangle_node = True
            elif key == "rotate" and separator:
                style.rotate_degrees = self._eval_expr(value, picture.symbols)
            elif key == "inner sep" and separator:
                separation = self._parse_length(value, picture.symbols).pt
                style.inner_xsep_pt = separation
                style.inner_ysep_pt = separation
            elif key == "inner xsep" and separator:
                style.inner_xsep_pt = self._parse_length(value, picture.symbols).pt
            elif key == "inner ysep" and separator:
                style.inner_ysep_pt = self._parse_length(value, picture.symbols).pt
            elif key == "dashed":
                style.dash_pattern_pt = (3.0, 3.0)
            elif key == "densely dashed":
                style.dash_pattern_pt = (2.0, 2.0)
            elif key == "dash pattern" and separator:
                dash_match = re.fullmatch(
                    r"on\s+([^\s]+)\s+off\s+([^\s]+)", value
                )
                if not dash_match:
                    raise TikzNativeError(f"unsupported dash pattern: {value}")
                style.dash_pattern_pt = (
                    self._parse_length(dash_match.group(1), picture.symbols).pt,
                    self._parse_length(dash_match.group(2), picture.symbols).pt,
                )
            elif key.startswith("-{Stealth") or option.startswith("-{Stealth"):
                style.arrow_tip = "Stealth"
                length_match = re.search(r"length=([^,\]]+)", option)
                width_match = re.search(r"width=([^,\]]+)", option)
                if length_match:
                    style.arrow_length_pt = self._parse_length(
                        length_match.group(1), picture.symbols
                    ).pt
                if width_match:
                    style.arrow_width_pt = self._parse_length(
                        width_match.group(1), picture.symbols
                    ).pt
            elif key in {"line cap", "line join"} and separator:
                setattr(style, key.replace(" ", "_"), value)
            elif key in {
                "angle radius",
                "angle eccentricity",
                "pos",
                "anchor",
                "above",
                "below",
                "left",
                "right",
                "above left",
                "above right",
                "below left",
                "below right",
                "midway",
                "sloped",
                "circle",
                "inner sep",
                "inner xsep",
                "inner ysep",
                "outer sep",
                "text opacity",
                "anglemark",
                "pic text options",
            } or option.startswith('"'):
                continue
            elif self._looks_like_color(option):
                resolved = self._resolve_color(option)
                if command == "fill":
                    style.fill_color = resolved
                else:
                    style.draw_color = resolved
            else:
                raise TikzNativeError(f"unsupported style option: {option}")
        if command == "fill" and style.fill_color is None:
            style.fill_color = defaults.draw_color or "#20242A"
        return style

    def _expand_named_style_options(
        self,
        options: str,
        picture: PictureSpec,
    ) -> list[str]:
        pending = _split_top_level(options)
        expanded: list[str] = []
        expansion_count = 0
        while pending:
            option = pending.pop(0).strip()
            key, separator, _value = option.partition("=")
            if not separator and key.strip() in picture.named_styles:
                expansion_count += 1
                if expansion_count > 64:
                    raise TikzNativeError("named TikZ style expansion appears recursive")
                pending = _split_top_level(picture.named_styles[key.strip()]) + pending
            elif option:
                expanded.append(option)
        return expanded

    def _looks_like_color(self, value: str) -> bool:
        base = value.split("!", 1)[0]
        return base in self.colors

    def _resolve_color(self, expression: str) -> str:
        parts = expression.strip().split("!")
        base = parts[0]
        if base not in self.colors:
            raise TikzNativeError(f"unknown color: {base}")
        color = self.colors[base]
        if len(parts) == 1:
            return color
        try:
            percent = float(parts[1]) / 100.0
        except ValueError as error:
            raise TikzNativeError(f"invalid xcolor percentage: {expression}") from error
        other_name = parts[2] if len(parts) >= 3 and parts[2] else "white"
        if other_name not in self.colors:
            raise TikzNativeError(f"unknown xcolor mix target: {other_name}")
        return self._mix_hex(color, self.colors[other_name], percent)

    @staticmethod
    def _mix_hex(first: str, second: str, first_fraction: float) -> str:
        first_rgb = tuple(int(first[index : index + 2], 16) for index in (1, 3, 5))
        second_rgb = tuple(int(second[index : index + 2], 16) for index in (1, 3, 5))
        mixed = tuple(
            round(first_fraction * a + (1.0 - first_fraction) * b)
            for a, b in zip(first_rgb, second_rgb)
        )
        return "#" + "".join(f"{value:02X}" for value in mixed)

    def _compile_path_command(
        self,
        command: str,
        statement: str,
        picture: PictureSpec,
        defaults: StyleSpec,
        z_index: int,
        source_line: int,
    ) -> list[ObjectSpec]:
        options, path = self._extract_options(statement, command)
        style = self._parse_style(options, command, defaults, picture)
        path_without_nodes, node_specs = self._extract_inline_nodes(path)

        ellipse_match = re.fullmatch(
            r"(.+?)\s+ellipse\s*\[([^\]]+)\]\s*",
            path_without_nodes,
            flags=re.DOTALL,
        )
        if ellipse_match:
            center = self._parse_coord(ellipse_match.group(1).strip(), picture)
            radii = dict(
                part.split("=", 1) for part in _split_top_level(ellipse_match.group(2))
            )
            rx = self._eval_expr(radii["x radius"], picture.symbols)
            ry = self._eval_expr(radii["y radius"], picture.symbols)
            objects = [
                self._object(
                    self._semantic_id("ellipse", [center.name]),
                    "ellipse",
                    {
                        "center": center.xy,
                        "center_name": center.name,
                        "rx": rx,
                        "ry": ry,
                    },
                    style,
                    z_index,
                    source_line,
                    statement,
                )
            ]
            objects.extend(
                self._compile_inline_nodes(
                    node_specs,
                    [(center, *ellipse_match.span(1))],
                    path_without_nodes,
                    picture,
                    style,
                    z_index + 1,
                    source_line,
                    statement,
                )
            )
            return objects

        circle_match = re.fullmatch(
            r"(.+?)\s+circle\s*\(([^)]+)\)\s*",
            path_without_nodes,
            flags=re.DOTALL,
        )
        if circle_match:
            center = self._parse_coord(circle_match.group(1).strip(), picture)
            radius_raw = circle_match.group(2).strip()
            if re.search(r"(?:pt|mm|cm)$", radius_raw):
                radius_pt = self._parse_length(radius_raw, picture.symbols).pt
                kind = "dot" if command == "fill" else "circle"
                geometry = {
                    "center": center.xy,
                    "center_name": center.name,
                    "radius_pt": radius_pt,
                }
            else:
                radius = self._eval_expr(radius_raw, picture.symbols)
                kind = "circle"
                geometry = {
                    "center": center.xy,
                    "center_name": center.name,
                    "radius": radius,
                }
            objects = [
                self._object(
                    self._semantic_id(kind, [center.name]),
                    kind,
                    geometry,
                    style,
                    z_index,
                    source_line,
                    statement,
                )
            ]
            objects.extend(
                self._compile_inline_nodes(
                    node_specs,
                    [(center, *circle_match.span(1))],
                    path_without_nodes,
                    picture,
                    style,
                    z_index + 1,
                    source_line,
                    statement,
                )
            )
            return objects

        coord_matches = self._find_path_coordinates(path_without_nodes, picture)
        if len(coord_matches) < 2:
            if coord_matches and node_specs:
                return self._compile_inline_nodes(
                    node_specs,
                    coord_matches,
                    path_without_nodes,
                    picture,
                    style,
                    z_index + 1,
                    source_line,
                    statement,
                )
            if command == "fill" and len(coord_matches) == 1:
                picture.warnings.append(
                    "empty one-point fill has no native geometry and was ignored"
                )
                return []
            raise TikzNativeError("draw/fill path must contain at least two coordinates")
        coordinate_groups = self._path_coordinate_groups(
            path_without_nodes,
            coord_matches,
        )
        objects: list[ObjectSpec] = []
        points = [item[0].xy for item in coord_matches]
        names = [item[0].name for item in coord_matches]
        if command in {"fill", "filldraw"}:
            if len(coordinate_groups) != 1 or not coordinate_groups[0][1]:
                raise TikzNativeError("open fill paths are not supported")
            indices = coordinate_groups[0][0]
            objects.append(
                self._object(
                    self._semantic_id("fill", [names[index] for index in indices]),
                    "polygon",
                    {
                        "points": [points[index] for index in indices],
                        "point_names": [names[index] for index in indices],
                        "closed": True,
                    },
                    style,
                    z_index,
                    source_line,
                    statement,
                )
            )
        else:
            segment_pairs: list[tuple[int, int]] = []
            for indices, is_closed in coordinate_groups:
                segment_pairs.extend(zip(indices, indices[1:]))
                if is_closed and len(indices) > 2:
                    segment_pairs.append((indices[-1], indices[0]))
            for pair_index, (segment_index, next_index) in enumerate(segment_pairs):
                segment_style = StyleSpec(**asdict(style))
                kind = (
                    "arrow"
                    if style.arrow_tip and pair_index == len(segment_pairs) - 1
                    else "line"
                )
                objects.append(
                    self._object(
                        self._semantic_id(
                            kind,
                            [names[segment_index], names[next_index]],
                        ),
                        kind,
                        {
                            "start": points[segment_index],
                            "end": points[next_index],
                            "start_name": names[segment_index],
                            "end_name": names[next_index],
                        },
                        segment_style,
                        z_index,
                        source_line,
                        statement,
                    )
                )
        objects.extend(
            self._compile_inline_nodes(
                node_specs,
                coord_matches,
                path_without_nodes,
                picture,
                style,
                z_index + 1,
                source_line,
                statement,
            )
        )
        return objects

    @staticmethod
    def _path_coordinate_groups(
        path: str,
        coordinates: list[tuple[_CoordValue, int, int]],
    ) -> list[tuple[list[int], bool]]:
        """Split TikZ subpaths so disconnected segments are never bridged."""

        if not coordinates:
            return []
        groups: list[tuple[list[int], bool]] = []
        current = [0]
        for index in range(1, len(coordinates)):
            between = path[coordinates[index - 1][2] : coordinates[index][1]]
            if re.search(r"--", between) and not re.search(r"\bcycle\b", between):
                current.append(index)
                continue
            groups.append((current, bool(re.search(r"--\s*cycle\b", between))))
            current = [index]
        tail = path[coordinates[-1][2] :]
        groups.append((current, bool(re.search(r"--\s*cycle\b", tail))))
        return groups

    def _find_path_coordinates(
        self,
        path: str,
        picture: PictureSpec,
    ) -> list[tuple[_CoordValue, int, int]]:
        results: list[tuple[_CoordValue, int, int]] = []
        index = 0
        while index < len(path):
            if path.startswith("$(", index):
                end = path.find(")$", index)
                if end < 0:
                    raise TikzNativeError("unclosed calc coordinate")
                raw = path[index : end + 2]
                results.append((self._parse_coord(raw, picture), index, end + 2))
                index = end + 2
            elif path[index] == "(":
                depth = 0
                end = index
                while end < len(path):
                    if path[end] == "(":
                        depth += 1
                    elif path[end] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    end += 1
                if depth != 0:
                    raise TikzNativeError("unclosed path coordinate")
                raw = path[index : end + 1]
                results.append((self._parse_coord(raw, picture), index, end + 1))
                index = end + 1
            else:
                index += 1
        return results

    @staticmethod
    def _extract_inline_nodes(path: str) -> tuple[str, list[tuple[int, str, str]]]:
        node_re = re.compile(r"\\?node(?:\[([^\]]*)\])?\s*\{")
        cursor = 0
        output: list[str] = []
        nodes: list[tuple[int, str, str]] = []
        while match := node_re.search(path, cursor):
            output.append(path[cursor : match.start()])
            content, end = _extract_balanced(path, match.end() - 1)
            placeholder_position = sum(len(part) for part in output)
            nodes.append((placeholder_position, match.group(1) or "", content))
            cursor = end
        output.append(path[cursor:])
        return "".join(output), nodes

    def _compile_inline_nodes(
        self,
        nodes: list[tuple[int, str, str]],
        coords: list[tuple[_CoordValue, int, int]],
        path_without_nodes: str,
        picture: PictureSpec,
        defaults: StyleSpec,
        z_index: int,
        source_line: int,
        raw: str,
    ) -> list[ObjectSpec]:
        objects: list[ObjectSpec] = []
        for position, options, content in nodes:
            if not content.strip():
                continue
            placement = self._parse_label_placement(options, picture)
            previous_index = max(
                (index for index, item in enumerate(coords) if item[2] <= position),
                default=0,
            )
            next_index = min(previous_index + 1, len(coords) - 1)
            between_previous_and_node = path_without_nodes[
                coords[previous_index][2] : position
            ]
            before_next_coordinate = (
                next_index > previous_index and position <= coords[next_index][1]
            )
            follows_segment_operator = bool(
                re.search(r"--\s*$", between_previous_and_node)
            )
            is_path = (
                placement.path_pos is not None
                or placement.sloped
                or (before_next_coordinate and follows_segment_operator)
            )
            if is_path and previous_index == next_index and previous_index > 0:
                previous_index -= 1
            if is_path:
                start = coords[previous_index][0]
                end = coords[next_index][0]
                pos = placement.path_pos if placement.path_pos is not None else 0.5
                geometry = {
                    "start": start.xy,
                    "end": end.xy,
                    "start_name": start.name,
                    "end_name": end.name,
                    "pos": pos,
                }
                object_id = self._semantic_id(
                    "label_path",
                    [start.name, end.name, _slug(content)],
                )
            else:
                point = coords[previous_index][0]
                geometry = {"at": point.xy, "at_name": point.name}
                object_id = self._semantic_id("label", [point.name, _slug(content)])
            style = self._parse_style(options, "node", defaults, picture)
            objects.append(
                self._object(
                    object_id,
                    "path_label" if is_path else "label",
                    geometry,
                    style,
                    z_index,
                    source_line,
                    raw,
                    label=content,
                    placement=placement,
                )
            )
        return objects

    def _compile_standalone_node(
        self,
        statement: str,
        picture: PictureSpec,
        defaults: StyleSpec,
        z_index: int,
        source_line: int,
    ) -> ObjectSpec:
        options, rest = self._extract_options(statement, "node")
        at_match = re.match(r"at\s+", rest)
        if not at_match:
            raise TikzNativeError("standalone node requires 'at'")
        rest = rest[at_match.end() :].strip()
        if rest.startswith("$("):
            coordinate_end = rest.find(")$")
            if coordinate_end < 0:
                raise TikzNativeError("unclosed calc coordinate in node")
            coordinate_end += 2
        elif rest.startswith("("):
            depth = 0
            coordinate_end = -1
            for index, char in enumerate(rest):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        coordinate_end = index + 1
                        break
            if coordinate_end < 0:
                raise TikzNativeError("unclosed coordinate in node")
        else:
            raise TikzNativeError("node coordinate must be parenthesized")
        coordinate_raw = rest[:coordinate_end].strip()
        remainder = rest[coordinate_end:].strip()
        if remainder.startswith("["):
            trailing_options, options_end = _extract_balanced(
                remainder,
                0,
                "[",
                "]",
            )
            options = ",".join(
                part for part in (options, trailing_options) if part.strip()
            )
            remainder = remainder[options_end:].strip()
        if not remainder.startswith("{"):
            raise TikzNativeError("node content missing")
        content, content_end = _extract_balanced(remainder, 0)
        if remainder[content_end:].strip():
            raise TikzNativeError("unexpected tokens after node content")
        coordinate = self._parse_coord(coordinate_raw, picture)
        expanded_options = self._expand_named_style_options(options, picture)
        if not content.strip() and "circle" in expanded_options:
            inner_sep_pt = 0.8
            for option in expanded_options:
                key, separator, value = option.partition("=")
                if key.strip() == "inner sep" and separator:
                    inner_sep_pt = self._parse_length(value, picture.symbols).pt
            style = self._parse_style(
                ",".join(expanded_options),
                "fill",
                defaults,
                picture,
            )
            return self._object(
                self._semantic_id("dot", [coordinate.name]),
                "dot",
                {
                    "center": coordinate.xy,
                    "center_name": coordinate.name,
                    "radius_pt": inner_sep_pt,
                },
                style,
                z_index,
                source_line,
                statement,
            )
        placement = self._parse_label_placement(options, picture)
        style = self._parse_style(options, "node", defaults, picture)
        return self._object(
            self._semantic_id("label", [coordinate.name, _slug(content)]),
            "label",
            {"at": coordinate.xy, "at_name": coordinate.name},
            style,
            z_index,
            source_line,
            statement,
            label=content,
            placement=placement,
        )

    def _parse_label_placement(
        self,
        options: str,
        picture: PictureSpec,
    ) -> LabelPlacement:
        placement = LabelPlacement()
        for option in _split_top_level(options):
            key, separator, value = option.partition("=")
            key = key.strip()
            value = value.strip()
            if key in self._direction_keys:
                amount = 0.0
                if separator and value:
                    amount = self._parse_length(value, picture.symbols).pt
                if key == "above":
                    placement.anchor = "south"
                    placement.dy_pt += amount
                elif key == "below":
                    placement.anchor = "north"
                    placement.dy_pt -= amount
                elif key == "left":
                    placement.anchor = "east"
                    placement.dx_pt -= amount
                elif key == "right":
                    placement.anchor = "west"
                    placement.dx_pt += amount
                elif key == "above left":
                    placement.anchor = "south east"
                    placement.dx_pt -= amount
                    placement.dy_pt += amount
                elif key == "above right":
                    placement.anchor = "south west"
                    placement.dx_pt += amount
                    placement.dy_pt += amount
                elif key == "below left":
                    placement.anchor = "north east"
                    placement.dx_pt -= amount
                    placement.dy_pt -= amount
                elif key == "below right":
                    placement.anchor = "north west"
                    placement.dx_pt += amount
                    placement.dy_pt -= amount
            elif key == "anchor" and separator:
                if value not in self._anchor_map:
                    raise TikzNativeError(f"unsupported node anchor: {value}")
                placement.anchor = value
            elif key == "pos" and separator:
                placement.path_pos = self._eval_expr(value, picture.symbols)
            elif key == "midway":
                placement.path_pos = 0.5
            elif key == "sloped":
                placement.sloped = True
            elif key == "font" and separator:
                placement.font_command = value
        return placement

    def _compile_pic(
        self,
        statement: str,
        picture: PictureSpec,
        defaults: StyleSpec,
        z_index: int,
        source_line: int,
    ) -> list[ObjectSpec]:
        options, rest = self._extract_options(statement, "pic")
        if not rest.startswith("{"):
            raise TikzNativeError("pic body must be braced")
        body, end = _extract_balanced(rest, 0)
        if rest[end:].strip():
            raise TikzNativeError("unexpected tokens after pic")
        match = re.fullmatch(
            r"(right angle|angle)\s*=\s*([^\-]+)--([^\-]+)--(.+)",
            body.strip(),
        )
        if not match:
            raise TikzNativeError(f"unsupported pic body: {body}")
        kind_name = match.group(1)
        first_name, vertex_name, third_name = (
            match.group(2).strip(),
            match.group(3).strip(),
            match.group(4).strip(),
        )
        first = self._lookup_named(first_name, picture)
        vertex = self._lookup_named(vertex_name, picture)
        third = self._lookup_named(third_name, picture)
        style = self._parse_style(options, "draw", defaults, picture)
        radius_pt = 10.0
        eccentricity = 1.0
        label: str | None = None
        label_placement = LabelPlacement(font_command=style.font_command)
        label_options = ""
        uses_anglemark = False
        for option in _split_top_level(options):
            key, separator, value = option.partition("=")
            key = key.strip()
            value = value.strip()
            if key == "angle radius" and separator:
                radius_pt = self._parse_length(value, picture.symbols).pt
            elif key == "angle eccentricity" and separator:
                eccentricity = self._eval_expr(value, picture.symbols)
            elif key == "anglemark" and separator:
                uses_anglemark = True
                angle_label = self._unbrace_option_value(value).strip()
                label = f"${angle_label}$" if angle_label else None
                # The shared handout anglemark style sets the pic text at the
                # vertex; explicit pic-text pt shifts are applied afterward.
                eccentricity = 0.0
            elif key == "pic text options" and separator:
                label_options = self._unbrace_option_value(value)
                label_placement = self._parse_label_placement(
                    label_options,
                    picture,
                )
            elif option.startswith('"') and option.endswith('"'):
                label = option[1:-1]
        if uses_anglemark:
            # ``math-handout-common.sty`` defines this semantic style.  It is
            # source-level project vocabulary, not an arbitrary TikZ fallback.
            style.fill_color = self._resolve_color("cyan!35")
            style.fill_opacity = 0.5
            eccentricity = 0.0
        geometry = {
            "first": first,
            "vertex": vertex,
            "third": third,
            "first_name": first_name,
            "vertex_name": vertex_name,
            "third_name": third_name,
            "radius_pt": radius_pt,
            "eccentricity": eccentricity,
        }
        result = [
            self._object(
                self._semantic_id(
                    "right_angle" if kind_name == "right angle" else "angle",
                    [first_name, vertex_name, third_name],
                ),
                "right_angle" if kind_name == "right angle" else "angle",
                geometry,
                style,
                z_index,
                source_line,
                statement,
            )
        ]
        if label is not None:
            label_style = StyleSpec(**asdict(style))
            if label_options:
                label_style = self._parse_style(
                    label_options,
                    "node",
                    label_style,
                    picture,
                )
            # ``anglemark`` fills the sector, not the pic-text node.
            label_style.fill_color = None
            label_style.fill_opacity = None
            label_style.rectangle_node = False
            result.append(
                self._object(
                    self._semantic_id(
                        "label_angle",
                        [first_name, vertex_name, third_name, _slug(label)],
                    ),
                    "angle_label",
                    geometry,
                    label_style,
                    z_index + 1,
                    source_line,
                    statement,
                    label=label,
                    placement=label_placement,
                )
            )
        return result

    def _semantic_id(self, prefix: str, parts: Iterable[str | None]) -> str:
        core = ".".join(part for part in parts if part)
        base = f"{prefix}.{core}" if core else prefix
        count = self._object_ids.get(base, 0) + 1
        self._object_ids[base] = count
        return base if count == 1 else f"{base}.{count}"

    @staticmethod
    def _object(
        object_id: str,
        kind: str,
        geometry: dict[str, Any],
        style: StyleSpec,
        z_index: int,
        source_line: int,
        raw: str,
        label: str | None = None,
        placement: LabelPlacement | None = None,
    ) -> ObjectSpec:
        return ObjectSpec(
            id=object_id,
            kind=kind,
            geometry=geometry,
            style=style,
            z_index=z_index,
            source_line=source_line,
            raw=re.sub(r"\s+", " ", raw).strip(),
            label=label,
            placement=placement,
        )


def compile_document(
    source_path: str | Path | None = None,
    *,
    source_text: str | None = None,
    entry_macro: str | None = None,
) -> DocumentSpec:
    return TikzNativeCompiler(
        source_path,
        source_text=source_text,
        entry_macro=entry_macro,
    ).compile()
