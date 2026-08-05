from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import atan2, cos, sin
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import numpy as np
from PIL import Image
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    TAU,
    UP,
    Arc,
    CapStyleType,
    Circle,
    Dot,
    Ellipse,
    Line,
    LineJointType,
    MathTex,
    Mobject,
    Polygon,
    Rectangle,
    RightAngle,
    StealthTip,
    Tex,
    TexTemplate,
    VGroup,
)

from .compiler import ObjectSpec, PictureSpec, StyleSpec, TEX_PT_PER_CM


DEFAULT_TEX_TEMPLATE = TexTemplate(
    tex_compiler="xelatex",
    output_format=".xdv",
    documentclass=r"\documentclass[preview,11pt]{standalone}",
    preamble=r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{fontspec}
\usepackage{xeCJK}
\usepackage{unicode-math}
\usepackage{tikz}
\setCJKmainfont[
  Path=/Users/leocyan/Library/Fonts/,
  BoldFont=simhei.ttf,
  ItalicFont=simfang.ttf,
  FakeBold=2
]{simsun.ttc}
\setCJKsansfont[
  Path=/Users/leocyan/Library/Fonts/,
  BoldFont=simhei.ttf,
  FakeBold=2
]{simsun.ttc}
\setmathfont[
  Path=/Users/leocyan/Library/Fonts/,
  FakeBold=2
]{latinmodern-math.otf}
\setmainfont[
  Path=/Users/leocyan/Library/Fonts/,
  FakeBold=2,
  BoldFont=lmroman10-bold.otf,
  ItalicFont=lmroman10-italic.otf,
  BoldItalicFont=lmroman10-bolditalic.otf
]{lmroman10-regular.otf}
\newlength{\manimwhiteoutlineoffset}
\newlength{\manimwhiteoutlineinneroffset}
\newcommand{\manimwhiteoutlinedraw}[2]{%
  \tikz[baseline=(manimoutlinebase.base)]{%
    \node[inner sep=0pt,outer sep=0pt,text opacity=0]
      (manimoutlinebase) {$#1#2$};
    \foreach \manimoutlineangle in {0,15,...,345}{%
      \node[inner sep=0pt,outer sep=0pt,text=white,
        shift={(\manimoutlineangle:\manimwhiteoutlineoffset)}]
        at (manimoutlinebase.center) {$#1#2$};
      \node[inner sep=0pt,outer sep=0pt,text=white,
        shift={(\manimoutlineangle:\manimwhiteoutlineinneroffset)}]
        at (manimoutlinebase.center) {$#1#2$};
    }
    \node[inner sep=0pt,outer sep=0pt,text=black]
      at (manimoutlinebase.center) {$#1#2$};
  }%
}
\newcommand{\mathWhiteOutline}[2][0.55]{%
  \begingroup
  \setlength{\manimwhiteoutlineoffset}{#1pt}%
  \setlength{\manimwhiteoutlineinneroffset}{0.55\manimwhiteoutlineoffset}%
  \ensuremath{\mathchoice
    {\manimwhiteoutlinedraw{\displaystyle}{#2}}%
    {\manimwhiteoutlinedraw{\textstyle}{#2}}%
    {\manimwhiteoutlinedraw{\scriptstyle}{#2}}%
    {\manimwhiteoutlinedraw{\scriptscriptstyle}{#2}}}%
  \endgroup
}
""",
)

# Calibrated with ``\rule{1cm}{1pt}``: at this Manim font size one TeX
# centimetre occupies one scene unit. Multiplying by ``scene_unit_per_cm``
# preserves the physical 11 pt body font relative to TikZ coordinates.
MANIM_FONT_SIZE_PER_TEX_CM = 33.86666666666667

# Cairo's low-resolution antialiasing makes a nominally physical Manim stroke
# visibly lighter than the same TikZ stroke at 60 px/cm.  A 3.8 px/pt geometry
# calibration matches the source's effective ink weight much more closely.
# Text halos are a separate visual effect and retain their previous 2.15
# calibration so the white-outline label regression does not drift.
GEOMETRY_STROKE_WIDTH_PER_PT = 3.8
LABEL_OUTLINE_STROKE_WIDTH_PER_PT = 2.15

# TikZ's default ``inner sep`` is ``0.3333em``.  At the source document's
# 10.95 pt normal font this is 3.6496 pt.  Ordinary labels still need the
# empirical visible-ink calibration below, but a ``\mathWhiteOutline`` label
# carries its complete logical TeX box and must therefore use the real TikZ
# padding on both axes.
TIKZ_DEFAULT_INNER_SEP_PT = 3.6496

FONT_SIZE_COMMAND_RE = re.compile(
    r"^(\\(?:tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge))"
    r"(?=\s|\\|\$|\{)"
)

INLINE_MATH_RE = re.compile(
    r"(?<!\\)\$(?!\$)(.*?)(?<!\\)\$",
    flags=re.DOTALL,
)

PAREN_MATH_RE = re.compile(
    r"\\\((.*?)\\\)",
    flags=re.DOTALL,
)

WHITE_OUTLINE_PROBE_DPI = 600
WHITE_OUTLINE_NODE_OPTION_RE = re.compile(
    r"^(?:"
    r"above(?:\s+(?:left|right))?|"
    r"below(?:\s+(?:left|right))?|"
    r"left|right|anchor|"
    r"inner(?:\s+[xy])?sep|outer(?:\s+[xy])?sep"
    r")(?:\s*=|$)"
)


ANCHOR_TO_EDGE = {
    "center": ORIGIN,
    "north": UP,
    "south": DOWN,
    "west": LEFT,
    "east": RIGHT,
    "north west": UP + LEFT,
    "north east": UP + RIGHT,
    "south west": DOWN + LEFT,
    "south east": DOWN + RIGHT,
}


@dataclass
class NativeFigure:
    picture: PictureSpec
    objects: dict[str, Mobject]
    group: VGroup
    warnings: list[str]


class NativeManimRenderer:
    """Build only semantic Manim primitives from a :class:`PictureSpec`."""

    _white_outline_metric_cache: dict[
        str,
        tuple[float, float, float, float],
    ] = {}

    def __init__(
        self,
        *,
        scene_unit_per_cm: float = 1.0,
        base_font_size: float | None = None,
        stroke_width_per_pt: float = GEOMETRY_STROKE_WIDTH_PER_PT,
        label_outline_stroke_width_per_pt: float = (
            LABEL_OUTLINE_STROKE_WIDTH_PER_PT
        ),
        tex_template: TexTemplate = DEFAULT_TEX_TEMPLATE,
    ) -> None:
        self.unit = scene_unit_per_cm
        self.base_font_size = (
            MANIM_FONT_SIZE_PER_TEX_CM * scene_unit_per_cm
            if base_font_size is None
            else base_font_size
        )
        self.stroke_width_per_pt = stroke_width_per_pt
        self.label_outline_stroke_width_per_pt = label_outline_stroke_width_per_pt
        self.tex_template = tex_template

    @property
    def pt(self) -> float:
        return self.unit / TEX_PT_PER_CM

    def render(self, picture: PictureSpec) -> NativeFigure:
        objects: dict[str, Mobject] = {}
        warnings = list(picture.warnings)
        for spec in picture.objects:
            try:
                mobject = self._build(spec, picture)
            except Exception as error:
                raise RuntimeError(
                    f"Failed to build picture {picture.index} object {spec.id} "
                    f"({spec.kind}): {error}"
                ) from error
            mobject.set_z_index(spec.z_index)
            objects[spec.id] = mobject
        return NativeFigure(picture, objects, VGroup(*objects.values()), warnings)

    def point(self, value: tuple[float, float], picture: PictureSpec) -> np.ndarray:
        return self.unit * picture.scale * (
            float(value[0]) * RIGHT + float(value[1]) * UP
        )

    def _stroke_width(self, style: StyleSpec) -> float:
        return style.line_width_pt * self.stroke_width_per_pt

    @staticmethod
    def _opacity(style: StyleSpec, layer: str) -> float:
        specific = style.fill_opacity if layer == "fill" else style.draw_opacity
        return style.opacity * (1.0 if specific is None else specific)

    @staticmethod
    def _cap_style(value: str) -> CapStyleType:
        return {
            "round": CapStyleType.ROUND,
            "butt": CapStyleType.BUTT,
            "square": CapStyleType.SQUARE,
        }.get(value, CapStyleType.AUTO)

    @staticmethod
    def _joint_style(value: str) -> LineJointType:
        return {
            "round": LineJointType.ROUND,
            "bevel": LineJointType.BEVEL,
            "miter": LineJointType.MITER,
        }.get(value, LineJointType.AUTO)

    def _line_kwargs(self, style: StyleSpec) -> dict[str, Any]:
        return {
            "color": style.draw_color or "#20242A",
            "stroke_width": self._stroke_width(style),
            "stroke_opacity": self._opacity(style, "draw"),
            "cap_style": self._cap_style(style.line_cap),
            "joint_type": self._joint_style(style.line_join),
        }

    def _build(self, spec: ObjectSpec, picture: PictureSpec) -> Mobject:
        builders = {
            "line": self._build_line,
            "arrow": self._build_arrow,
            "polygon": self._build_polygon,
            "ellipse": self._build_ellipse,
            "circle": self._build_circle,
            "dot": self._build_dot,
            "label": self._build_label,
            "path_label": self._build_path_label,
            "angle": self._build_angle,
            "angle_label": self._build_angle_label,
            "right_angle": self._build_right_angle,
        }
        if spec.kind not in builders:
            raise ValueError(f"No native builder for {spec.kind}")
        return builders[spec.kind](spec, picture)

    def _build_line(self, spec: ObjectSpec, picture: PictureSpec) -> Mobject:
        start = self.point(spec.geometry["start"], picture)
        end = self.point(spec.geometry["end"], picture)
        return self.native_line_from_points(start, end, spec.style)

    def native_line_from_points(
        self,
        start: np.ndarray,
        end: np.ndarray,
        style: StyleSpec,
    ) -> Mobject:
        """Build a solid or dashed native line from scene-space endpoints."""

        if style.dash_pattern_pt:
            on_pt, off_pt = style.dash_pattern_pt
            return self._native_dashes(start, end, on_pt, off_pt, style)
        return Line(start, end, **self._line_kwargs(style))

    def _native_dashes(
        self,
        start: np.ndarray,
        end: np.ndarray,
        on_pt: float,
        off_pt: float,
        style: StyleSpec,
    ) -> VGroup:
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length == 0:
            return VGroup()
        direction = vector / length
        on_length = max(on_pt * self.pt, 1e-6)
        off_length = max(off_pt * self.pt, 0.0)
        cursor = 0.0
        dashes: list[Line] = []
        while cursor < length - 1e-9:
            dash_end = min(cursor + on_length, length)
            dashes.append(
                Line(
                    start + cursor * direction,
                    start + dash_end * direction,
                    **self._line_kwargs(style),
                )
            )
            cursor += on_length + off_length
        return VGroup(*dashes)

    def _build_arrow(self, spec: ObjectSpec, picture: PictureSpec) -> Line:
        line = Line(
            self.point(spec.geometry["start"], picture),
            self.point(spec.geometry["end"], picture),
            **self._line_kwargs(spec.style),
        )
        tip_length = (
            spec.style.arrow_length_pt * self.pt
            if spec.style.arrow_length_pt is not None
            else 0.17
        )
        tip_width = (
            spec.style.arrow_width_pt * self.pt
            if spec.style.arrow_width_pt is not None
            else None
        )
        line.add_tip(
            tip_shape=StealthTip,
            tip_length=tip_length,
            tip_width=tip_width,
        )
        return line

    def _build_polygon(self, spec: ObjectSpec, picture: PictureSpec) -> Polygon:
        points = [self.point(tuple(point), picture) for point in spec.geometry["points"]]
        return self.native_polygon_from_points(points, spec.style)

    def native_polygon_from_points(
        self,
        points: list[np.ndarray],
        style: StyleSpec,
    ) -> Polygon:
        """Build a native polygon from scene-space vertices."""

        return Polygon(
            *points,
            fill_color=style.fill_color or "#20242A",
            fill_opacity=self._opacity(style, "fill"),
            stroke_color=style.draw_color or "#20242A",
            stroke_opacity=(
                self._opacity(style, "draw") if style.draw_color else 0.0
            ),
            stroke_width=self._stroke_width(style) if style.draw_color else 0.0,
            cap_style=self._cap_style(style.line_cap),
            joint_type=self._joint_style(style.line_join),
        )

    def _build_ellipse(self, spec: ObjectSpec, picture: PictureSpec) -> Ellipse:
        ellipse = Ellipse(
            width=2 * spec.geometry["rx"] * self.unit * picture.scale,
            height=2 * spec.geometry["ry"] * self.unit * picture.scale,
            fill_color=spec.style.fill_color or "#20242A",
            fill_opacity=(
                self._opacity(spec.style, "fill") if spec.style.fill_color else 0.0
            ),
            stroke_color=spec.style.draw_color or "#20242A",
            stroke_opacity=(
                self._opacity(spec.style, "draw") if spec.style.draw_color else 0.0
            ),
            stroke_width=self._stroke_width(spec.style),
            cap_style=self._cap_style(spec.style.line_cap),
            joint_type=self._joint_style(spec.style.line_join),
        )
        return ellipse.move_to(self.point(tuple(spec.geometry["center"]), picture))

    def _build_circle(self, spec: ObjectSpec, picture: PictureSpec) -> Circle:
        radius = (
            spec.geometry["radius"] * self.unit * picture.scale
            if "radius" in spec.geometry
            else spec.geometry["radius_pt"] * self.pt
        )
        circle = Circle(
            radius=radius,
            fill_color=spec.style.fill_color or "#20242A",
            fill_opacity=(
                self._opacity(spec.style, "fill") if spec.style.fill_color else 0.0
            ),
            stroke_color=spec.style.draw_color or "#20242A",
            stroke_opacity=(
                self._opacity(spec.style, "draw") if spec.style.draw_color else 0.0
            ),
            stroke_width=self._stroke_width(spec.style),
            cap_style=self._cap_style(spec.style.line_cap),
            joint_type=self._joint_style(spec.style.line_join),
        )
        return circle.move_to(self.point(tuple(spec.geometry["center"]), picture))

    def _build_dot(self, spec: ObjectSpec, picture: PictureSpec) -> Dot:
        radius = spec.geometry.get("radius_pt", 1.0) * self.pt
        color = spec.style.fill_color or spec.style.draw_color or "#20242A"
        return Dot(
            self.point(tuple(spec.geometry["center"]), picture),
            radius=radius,
            color=color,
            fill_opacity=self._opacity(spec.style, "fill"),
            stroke_width=0,
        )

    @staticmethod
    def _strip_outer_group(text: str) -> str:
        value = text.strip()
        if not (value.startswith("{") and value.endswith("}")):
            return value
        depth = 0
        for index, char in enumerate(value):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and index != len(value) - 1:
                    return value
        return value[1:-1].strip()

    def _make_label(self, raw: str, style: StyleSpec) -> Mobject:
        content = self._strip_outer_group(raw)
        metric_content = content
        content, white_outline_pt = self._strip_math_white_outline(content)
        font_commands: list[str] = []
        if style.font_command:
            font_commands.append(style.font_command.strip())
        if command_match := FONT_SIZE_COMMAND_RE.match(content):
            font_commands.append(command_match.group(1))
            content = content[command_match.end() :].strip()
            metric_command_match = FONT_SIZE_COMMAND_RE.match(metric_content)
            if metric_command_match:
                metric_content = metric_content[metric_command_match.end() :].strip()

        math_body = self._full_math_body(content)
        # The source macro explicitly draws its foreground node in black, so
        # an outlined label must not inherit a colored path's draw style (for
        # example the blue projection line carrying ``m'``).
        color = (
            "#000000"
            if white_outline_pt is not None
            else (style.draw_color or "#20242A")
        )
        if font_commands:
            command = " ".join(font_commands)
            if math_body is not None:
                math = self._ensure_displaystyle(math_body)
                render_source = rf"{{{command} ${math}$}}"
            else:
                render_source = rf"{{{command} {self._displaystyle_inline_math(content)}}}"
            label = Tex(
                render_source,
                tex_template=self.tex_template,
                font_size=self.base_font_size,
                color=color,
            )
        elif math_body is not None:
            render_source = self._ensure_displaystyle(math_body)
            label = MathTex(
                render_source,
                tex_template=self.tex_template,
                font_size=self.base_font_size,
                color=color,
            )
        else:
            render_source = self._displaystyle_inline_math(content)
            label = Tex(
                render_source,
                tex_template=self.tex_template,
                font_size=self.base_font_size,
                color=color,
            )
        if white_outline_pt is not None:
            label.set_stroke(
                color="#FFFFFF",
                width=(
                    2
                    * white_outline_pt
                    * self.label_outline_stroke_width_per_pt
                ),
                background=True,
            )
            metric_math_body = self._full_math_body(metric_content)
            if font_commands:
                command = " ".join(font_commands)
                if metric_math_body is not None:
                    metric_source = rf"{{{command} ${self._ensure_displaystyle(metric_math_body)}$}}"
                else:
                    metric_source = rf"{{{command} {self._displaystyle_inline_math(metric_content)}}}"
            elif metric_math_body is not None:
                metric_source = rf"${self._ensure_displaystyle(metric_math_body)}$"
            else:
                metric_source = self._displaystyle_inline_math(metric_content)
            (
                metric_width,
                metric_height,
                center_dx,
                center_dy,
            ) = self._white_outline_node_metrics(metric_source, style)
            metric_box = Rectangle(
                width=max(metric_width, 1e-6),
                height=max(metric_height, 1e-6),
                stroke_opacity=0.0,
                fill_opacity=0.0,
            ).move_to(label.get_center() + center_dx * RIGHT + center_dy * UP)
            label = VGroup(metric_box, label)
            # The metric box already represents the complete outer TikZ node,
            # including inner/outer sep and the unusual transform leakage from
            # directional placement keys into the nested outline tikzpicture.
            label._tikz_complete_node_box = True
            label._tikz_white_outline = True
        if style.fill_color is None and not style.rectangle_node:
            if style.rotate_degrees:
                label.rotate(style.rotate_degrees * TAU / 360.0)
            return label
        pad_x, pad_y = self._node_padding(label, style)
        background = Rectangle(
            width=label.width + 2 * pad_x,
            height=label.height + 2 * pad_y,
            stroke_color=style.draw_color or "#20242A",
            stroke_opacity=(
                self._opacity(style, "draw") if style.rectangle_node else 0.0
            ),
            stroke_width=(
                self._stroke_width(style) if style.rectangle_node else 0.0
            ),
            fill_color=style.fill_color or "#FFFFFF",
            fill_opacity=(
                self._opacity(style, "fill") if style.fill_color else 0.0
            ),
        ).move_to(label)
        group = VGroup(background, label)
        if style.rotate_degrees:
            group.rotate(style.rotate_degrees * TAU / 360.0)
        return group

    @staticmethod
    def _strip_math_white_outline(content: str) -> tuple[str, float | None]:
        """Map the handout readability macro to a native background stroke."""

        pattern = re.compile(r"\\mathWhiteOutline(?:\[([^\]]+)\])?\{")
        match = pattern.search(content)
        if match is None:
            return content, None
        depth = 1
        cursor = match.end()
        while cursor < len(content) and depth:
            if content[cursor] == "{":
                depth += 1
            elif content[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth:
            return content, None
        body = content[match.end() : cursor - 1]
        width_raw = match.group(1) or "0.55"
        try:
            width_pt = float(width_raw.strip())
        except ValueError:
            return content, None
        return content[: match.start()] + body + content[cursor:], width_pt

    @staticmethod
    def _full_math_body(content: str) -> str | None:
        stripped = content.strip()
        dollar_match = re.fullmatch(r"\$([^$]*)\$", stripped, flags=re.DOTALL)
        if dollar_match:
            return dollar_match.group(1)
        paren_match = re.fullmatch(r"\\\((.*?)\\\)", stripped, flags=re.DOTALL)
        if paren_match:
            return paren_match.group(1)
        return None

    @staticmethod
    def _white_outline_probe_options(style: StyleSpec) -> list[str]:
        """Keep node-box options whose transforms reach the nested TikZ."""

        options = [
            raw.strip()
            for raw in style.raw_options
            if WHITE_OUTLINE_NODE_OPTION_RE.match(raw.strip())
        ]
        has_inner_sep = any(
            raw.startswith(("inner sep", "inner xsep", "inner ysep"))
            for raw in options
        )
        if not has_inner_sep:
            if (
                style.inner_xsep_pt is not None
                and style.inner_ysep_pt is not None
                and style.inner_xsep_pt == style.inner_ysep_pt
            ):
                options.append(f"inner sep={style.inner_xsep_pt:g}pt")
            else:
                if style.inner_xsep_pt is not None:
                    options.append(f"inner xsep={style.inner_xsep_pt:g}pt")
                if style.inner_ysep_pt is not None:
                    options.append(f"inner ysep={style.inner_ysep_pt:g}pt")
        return options

    @staticmethod
    def _pixel_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
        rows, columns = np.where(mask)
        if not len(columns):
            raise RuntimeError("outlined-label metric probe produced an empty mask")
        return (
            int(columns.min()),
            int(rows.min()),
            int(columns.max()),
            int(rows.max()),
        )

    def _white_outline_node_metrics(
        self,
        content: str,
        style: StyleSpec,
    ) -> tuple[float, float, float, float]:
        r"""Measure the real outer node and black-ink offset of an outline.

        Directional TikZ keys such as ``above`` and ``right`` install canvas
        translations before the node text is boxed.  Because
        ``\mathWhiteOutline`` contains a nested ``tikzpicture``, those
        translations can enlarge its outer node and move the black formula
        within it.  A plain TeX hbox cannot observe that behavior.  This probe
        draws the exact node border in red, rasterizes it at high resolution,
        and caches four physical measurements in TeX points.
        """

        options = self._white_outline_probe_options(style)
        option_source = ",".join(options)
        cache_key = hashlib.sha256(
            (
                "white-outline-node-v2\n"
                + self.tex_template.body
                + self.tex_template.preamble
                + content
                + option_source
                + repr(WHITE_OUTLINE_PROBE_DPI)
            ).encode("utf-8")
        ).hexdigest()
        cached = self._white_outline_metric_cache.get(cache_key)
        if cached is None:
            cache_directory = (
                Path(tempfile.gettempdir())
                / "manim-tikz-white-outline-metrics-v2"
            )
            cache_path = cache_directory / f"{cache_key}.json"
            if cache_path.exists():
                values = json.loads(cache_path.read_text(encoding="utf-8"))
                cached = tuple(float(value) for value in values)
            else:
                cached = self._probe_white_outline_node_metrics(
                    content,
                    options,
                )
                cache_directory.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(cached) + "\n",
                    encoding="utf-8",
                )
            self._white_outline_metric_cache[cache_key] = cached

        scene_units_per_pt = (
            self.base_font_size / MANIM_FONT_SIZE_PER_TEX_CM / TEX_PT_PER_CM
        )
        return tuple(value * scene_units_per_pt for value in cached)

    def _probe_white_outline_node_metrics(
        self,
        content: str,
        options: list[str],
    ) -> tuple[float, float, float, float]:
        probe_options = ",".join(
            [*options, "draw=red", "line width=0.4pt"]
        )
        probe = (
            r"\begin{tikzpicture}"
            rf"\node[{probe_options}] at (0,0) {{{content}}};"
            r"\end{tikzpicture}"
        )
        tex_code = self.tex_template.get_texcode_for_expression(probe)
        with tempfile.TemporaryDirectory(
            prefix="manim-tikz-white-outline-metric-"
        ) as raw_dir:
            directory = Path(raw_dir)
            tex_path = directory / "metric.tex"
            tex_path.write_text(tex_code, encoding="utf-8")
            latex = subprocess.run(
                [
                    self.tex_template.tex_compiler,
                    "-interaction=batchmode",
                    "-halt-on-error",
                    tex_path.name,
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
            pdf_path = directory / "metric.pdf"
            if latex.returncode != 0 or not pdf_path.exists():
                log_path = directory / "metric.log"
                log = (
                    log_path.read_text(encoding="utf-8", errors="replace")
                    if log_path.exists()
                    else ""
                )
                diagnostic = (
                    latex.stdout + "\n" + latex.stderr + "\n" + log
                )[-4000:]
                raise RuntimeError(
                    "XeLaTeX could not render the outlined-label metric probe:\n"
                    + diagnostic
                )

            raster = subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    "1",
                    "-singlefile",
                    "-png",
                    "-r",
                    str(WHITE_OUTLINE_PROBE_DPI),
                    str(pdf_path),
                    str(directory / "metric"),
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
            png_path = directory / "metric.png"
            if raster.returncode != 0 or not png_path.exists():
                raise RuntimeError(
                    "pdftoppm could not rasterize the outlined-label metric "
                    "probe:\n" + (raster.stdout + "\n" + raster.stderr)[-4000:]
                )
            with Image.open(png_path) as raw_image:
                pixels = np.asarray(raw_image.convert("RGB"))

        red = (
            (pixels[:, :, 0] > 170)
            & (pixels[:, :, 1] < 150)
            & (pixels[:, :, 2] < 150)
        )
        black = pixels.max(axis=2) < 90
        red_left, red_top, red_right, red_bottom = self._pixel_bounds(red)
        ink_left, ink_top, ink_right, ink_bottom = self._pixel_bounds(black)
        node_center_x = (red_left + red_right) / 2
        node_center_y = (red_top + red_bottom) / 2
        ink_center_x = (ink_left + ink_right) / 2
        ink_center_y = (ink_top + ink_bottom) / 2
        tex_pt_per_pixel = TEX_PT_PER_CM * 2.54 / WHITE_OUTLINE_PROBE_DPI
        return (
            (red_right - red_left + 1) * tex_pt_per_pixel,
            (red_bottom - red_top + 1) * tex_pt_per_pixel,
            (node_center_x - ink_center_x) * tex_pt_per_pixel,
            (ink_center_y - node_center_y) * tex_pt_per_pixel,
        )

    @staticmethod
    def _ensure_displaystyle(math: str) -> str:
        if re.match(r"^\s*\\displaystyle(?:\s|$)", math):
            return math
        return rf"\displaystyle {math}"

    @classmethod
    def _displaystyle_inline_math(cls, content: str) -> str:
        content = PAREN_MATH_RE.sub(
            lambda match: f"${cls._ensure_displaystyle(match.group(1))}$",
            content,
        )
        return INLINE_MATH_RE.sub(
            lambda match: f"${cls._ensure_displaystyle(match.group(1))}$",
            content,
        )

    def _node_padding(
        self,
        label: Mobject,
        style: StyleSpec | None = None,
    ) -> tuple[float, float]:
        # An outlined label retains the complete TeX hbox, so its anchor must
        # use TikZ's actual 0.3333 em default inner sep.  Ordinary Manim labels
        # expose only visible SVG ink and retain the separately calibrated
        # asymmetric padding.
        if getattr(label, "_tikz_white_outline", False):
            default_x_pt = default_y_pt = TIKZ_DEFAULT_INNER_SEP_PT
        else:
            default_x_pt, default_y_pt = 4.7, 3.55
        if style is None:
            return default_x_pt * self.pt, default_y_pt * self.pt
        return (
            (
                default_x_pt
                if style.inner_xsep_pt is None
                else style.inner_xsep_pt
            )
            * self.pt,
            (
                default_y_pt
                if style.inner_ysep_pt is None
                else style.inner_ysep_pt
            )
            * self.pt,
        )

    def _outer_node_padding(self, label: Mobject, style: StyleSpec) -> tuple[float, float]:
        if getattr(label, "_tikz_complete_node_box", False):
            return 0.0, 0.0
        if style.fill_color is not None or style.rectangle_node:
            return 0.0, 0.0
        return self._node_padding(label, style)

    @staticmethod
    def _apply_node_shape_transform(
        label: Mobject,
        spec: ObjectSpec,
        picture: PictureSpec,
    ) -> Mobject:
        # Ordinary TikZ nodes ignore picture scale.  The explicit
        # ``transform shape`` key opts the node box and its text back into the
        # picture transform, so the authored scale must be applied here.
        if spec.style.transform_shape:
            label.scale(picture.scale)
        return label

    def _build_label(self, spec: ObjectSpec, picture: PictureSpec) -> Mobject:
        label = self._apply_node_shape_transform(
            self._make_label(spec.label or "", spec.style),
            spec,
            picture,
        )
        placement = spec.placement
        if placement is None:
            return label.move_to(self.point(tuple(spec.geometry["at"]), picture))
        target = self.point(tuple(spec.geometry["at"]), picture)
        target += self.pt * (placement.dx_pt * RIGHT + placement.dy_pt * UP)
        edge = ANCHOR_TO_EDGE[placement.anchor]
        pad_x, pad_y = self._outer_node_padding(label, spec.style)
        anchor_offset = np.array(
            [
                edge[0] * (label.width / 2 + pad_x),
                edge[1] * (label.height / 2 + pad_y),
                0.0,
            ]
        )
        return label.move_to(target - anchor_offset)

    def _build_path_label(self, spec: ObjectSpec, picture: PictureSpec) -> Mobject:
        label = self._apply_node_shape_transform(
            self._make_label(spec.label or "", spec.style),
            spec,
            picture,
        )
        placement = spec.placement
        if placement is None:
            raise ValueError("path label placement missing")
        start = self.point(tuple(spec.geometry["start"]), picture)
        end = self.point(tuple(spec.geometry["end"]), picture)
        vector = end - start
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError("path label cannot use a zero-length segment")
        tangent = vector / norm
        normal = np.array([-tangent[1], tangent[0], 0.0])
        base = start + float(spec.geometry["pos"]) * vector
        edge = ANCHOR_TO_EDGE[placement.anchor]
        pad_x, pad_y = self._outer_node_padding(label, spec.style)
        local_anchor = np.array(
            [
                edge[0] * (label.width / 2 + pad_x),
                edge[1] * (label.height / 2 + pad_y),
                0.0,
            ]
        )
        if not placement.sloped:
            target = base + self.pt * (
                placement.dx_pt * RIGHT + placement.dy_pt * UP
            )
            return label.move_to(target - local_anchor)

        target = base + self.pt * (
            placement.dx_pt * tangent + placement.dy_pt * normal
        )
        angle = atan2(tangent[1], tangent[0])
        display_angle = angle
        if display_angle > np.pi / 2 or display_angle < -np.pi / 2:
            display_angle += np.pi
        rotation = np.array(
            [
                [cos(display_angle), -sin(display_angle), 0.0],
                [sin(display_angle), cos(display_angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotated_anchor = rotation @ local_anchor
        label.rotate(display_angle, about_point=ORIGIN)
        return label.shift(target - rotated_anchor)

    def _angle_values(
        self, spec: ObjectSpec, picture: PictureSpec
    ) -> tuple[np.ndarray, float, float, float]:
        first = self.point(tuple(spec.geometry["first"]), picture)
        vertex = self.point(tuple(spec.geometry["vertex"]), picture)
        third = self.point(tuple(spec.geometry["third"]), picture)
        start = atan2((first - vertex)[1], (first - vertex)[0])
        end = atan2((third - vertex)[1], (third - vertex)[0])
        while end < start:
            end += TAU
        return vertex, start, end, spec.geometry["radius_pt"] * self.pt

    def _build_angle(self, spec: ObjectSpec, picture: PictureSpec) -> Arc:
        vertex, start, end, radius = self._angle_values(spec, picture)
        return Arc(
            radius=radius,
            start_angle=start,
            angle=end - start,
            arc_center=vertex,
            **self._line_kwargs(spec.style),
        )

    def _build_angle_label(self, spec: ObjectSpec, picture: PictureSpec) -> Mobject:
        label = self._make_label(spec.label or "", spec.style)
        vertex, start, end, radius = self._angle_values(spec, picture)
        midpoint = 0.5 * (start + end)
        target = vertex + spec.geometry["eccentricity"] * radius * (
            cos(midpoint) * RIGHT + sin(midpoint) * UP
        )
        return label.move_to(target)

    def _build_right_angle(self, spec: ObjectSpec, picture: PictureSpec) -> RightAngle:
        first = self.point(tuple(spec.geometry["first"]), picture)
        vertex = self.point(tuple(spec.geometry["vertex"]), picture)
        third = self.point(tuple(spec.geometry["third"]), picture)
        return self.native_right_angle_from_points(
            first,
            vertex,
            third,
            length=spec.geometry["radius_pt"] * self.pt,
            style=spec.style,
        )

    def native_right_angle_from_points(
        self,
        first: np.ndarray,
        vertex: np.ndarray,
        third: np.ndarray,
        *,
        length: float,
        style: StyleSpec,
    ) -> RightAngle:
        """Build a native right-angle marker from scene-space points."""

        line1 = Line(vertex, first)
        line2 = Line(vertex, third)
        return RightAngle(
            line1,
            line2,
            length=length,
            **self._line_kwargs(style),
        )
