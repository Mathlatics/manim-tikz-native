"""Shared v1 style contract for explicit planar curves in 3D."""

from __future__ import annotations

from math import isfinite
from typing import Protocol

import numpy as np


_DISPLAY_RELATIVE_TOLERANCE = float(np.sqrt(np.finfo(float).eps))
_DISPLAY_RANK_TOLERANCE = 64.0 * float(np.finfo(float).eps)
_MIN_NORMAL_DISPLAY_EXTENT = float(np.finfo(float).tiny)


class PlanarCurveStyleError(ValueError):
    """Raised when a style asks for unimplemented planar-curve semantics."""


class _PlanarCurveStyle(Protocol):
    draw_color: str | None
    fill_color: str | None
    opacity: float
    fill_opacity: float | None
    draw_opacity: float | None
    line_width_pt: float
    line_cap: str
    line_join: str
    dash_pattern_pt: tuple[float, float] | None
    arrow_tip: str | None
    arrow_length_pt: float | None
    arrow_width_pt: float | None
    font_command: str | None
    inner_xsep_pt: float | None
    inner_ysep_pt: float | None
    text_color: str | None
    node_border_color: str | None
    transform_shape: bool
    native_canvas_plane: str | None
    rectangle_node: bool
    rotate_degrees: float


def certify_planar_curve_display_scale(
    scene_unit_per_cm: object,
    picture_scale: object,
) -> float:
    """Return one finite positive renderer scale or fail before Manim sees it."""

    if isinstance(scene_unit_per_cm, bool) or isinstance(picture_scale, bool):
        raise PlanarCurveStyleError(
            "explicit 3D planar curve display scale must be finite and positive"
        )
    try:
        unit = float(scene_unit_per_cm)
        scale = float(picture_scale)
        result = unit * scale
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanarCurveStyleError(
            "explicit 3D planar curve display scale must be finite and positive"
        ) from exc
    if (
        not isfinite(unit)
        or not isfinite(scale)
        or not isfinite(result)
        or unit <= 0.0
        or scale <= 0.0
        or result <= 0.0
    ):
        raise PlanarCurveStyleError(
            "explicit 3D planar curve display scale must be finite and positive"
        )
    return result


def certify_planar_curve_affine_display(
    center: object,
    basis: object,
) -> None:
    """Certify that a rank-two affine curve remains representable in Manim.

    Finite multiplication alone is insufficient: a tiny display scale can
    underflow an axis, while a small radius translated to a very large screen
    center can disappear during ``center + axis``.  Check both rank and the
    actual representable cardinal displacements before constructing a Mobject.
    """

    try:
        center_array = np.asarray(center, dtype=float)
        basis_array = np.asarray(basis, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanarCurveStyleError(
            "explicit 3D planar curve display geometry must be finite"
        ) from exc
    if (
        center_array.ndim != 1
        or basis_array.shape != (center_array.size, 2)
        or center_array.size not in {2, 3}
        or not np.all(np.isfinite(center_array))
        or not np.all(np.isfinite(basis_array))
    ):
        raise PlanarCurveStyleError(
            "explicit 3D planar curve display geometry must be finite"
        )
    scale = float(np.max(np.abs(basis_array)))
    if not isfinite(scale) or scale < _MIN_NORMAL_DISPLAY_EXTENT:
        raise PlanarCurveStyleError(
            "explicit 3D planar curve display axes underflow the finite Manim range"
        )
    try:
        singular = np.linalg.svd(basis_array / scale, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise PlanarCurveStyleError(
            "explicit 3D planar curve display rank cannot be certified"
        ) from exc
    if (
        singular.shape != (2,)
        or not np.all(np.isfinite(singular))
        or float(singular[1])
        <= _DISPLAY_RANK_TOLERANCE * float(singular[0])
    ):
        raise PlanarCurveStyleError(
            "explicit 3D planar curve rank is not preserved in display space"
        )
    for axis in basis_array.T:
        axis_scale = float(np.max(np.abs(axis)))
        if not isfinite(axis_scale) or axis_scale < _MIN_NORMAL_DISPLAY_EXTENT:
            raise PlanarCurveStyleError(
                "explicit 3D planar curve display axis is not representable"
            )
        for direction in (-1.0, 1.0):
            endpoint = center_array + direction * axis
            actual = endpoint - center_array
            error = float(np.max(np.abs(actual - direction * axis)))
            relative_error = error / axis_scale
            if (
                not np.all(np.isfinite(endpoint))
                or not isfinite(relative_error)
                or relative_error > _DISPLAY_RELATIVE_TOLERANCE
            ):
                raise PlanarCurveStyleError(
                    "explicit 3D planar curve axis is not representable at its display center"
                )


def certify_planar_curve_display_segment(
    center: object,
    start_offset: object,
    end_offset: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a rank-one display segment and certify its authored extent.

    Keeping the offsets separate from the center matters at large coordinates:
    two finite endpoints can still lose one component of the intended segment
    when floating-point addition rounds ``center + offset``.
    """

    try:
        center_array = np.asarray(center, dtype=float)
        start_offset_array = np.asarray(start_offset, dtype=float)
        end_offset_array = np.asarray(end_offset, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanarCurveStyleError(
            "explicit 3D planar curve display segment must be finite"
        ) from exc
    if (
        center_array.ndim != 1
        or center_array.size not in {2, 3}
        or start_offset_array.shape != center_array.shape
        or end_offset_array.shape != center_array.shape
        or not np.all(np.isfinite(center_array))
        or not np.all(np.isfinite(start_offset_array))
        or not np.all(np.isfinite(end_offset_array))
    ):
        raise PlanarCurveStyleError(
            "explicit 3D planar curve display segment must be finite"
        )
    expected_extent = end_offset_array - start_offset_array
    extent_scale = float(np.max(np.abs(expected_extent)))
    if not isfinite(extent_scale) or extent_scale < _MIN_NORMAL_DISPLAY_EXTENT:
        raise PlanarCurveStyleError(
            "explicit 3D planar curve display segment has no representable finite extent"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        start = center_array + start_offset_array
        end = center_array + end_offset_array
    for endpoint, expected_offset in (
        (start, start_offset_array),
        (end, end_offset_array),
    ):
        offset_scale = max(
            extent_scale,
            float(np.max(np.abs(expected_offset))),
        )
        actual_offset = endpoint - center_array
        error = float(np.max(np.abs(actual_offset - expected_offset)))
        relative_error = error / offset_scale
        if (
            not np.all(np.isfinite(endpoint))
            or not isfinite(relative_error)
            or relative_error > _DISPLAY_RELATIVE_TOLERANCE
        ):
            raise PlanarCurveStyleError(
                "explicit 3D planar curve segment is not representable at its display center"
            )
    actual_extent = end - start
    extent_error = float(np.max(np.abs(actual_extent - expected_extent)))
    if (
        not np.all(np.isfinite(actual_extent))
        or not isfinite(extent_error)
        or extent_error / extent_scale > _DISPLAY_RELATIVE_TOLERANCE
    ):
        raise PlanarCurveStyleError(
            "explicit 3D planar curve segment extent is not representable at its display center"
        )
    return start, end


def validate_planar_curve_stroke_style(style: _PlanarCurveStyle) -> None:
    """Require the intentionally small static v1 rendering style.

    Fill would describe a planar disk rather than a curve.  Dashes require a
    separate screen-arclength and phase contract, especially when a circle is
    viewed edge-on.  Both therefore fail explicitly instead of inheriting a
    misleading two-dimensional implementation.
    """

    if not isinstance(style.draw_color, str) or not style.draw_color:
        raise PlanarCurveStyleError(
            "explicit 3D planar curves require a visible stroke"
        )
    if style.fill_color is not None or style.fill_opacity is not None:
        raise PlanarCurveStyleError(
            "explicit 3D planar curve v1 supports solid stroke only; fill is unsupported"
        )
    if style.dash_pattern_pt is not None:
        raise PlanarCurveStyleError(
            "explicit 3D planar curve v1 supports solid stroke only; dashed strokes are unsupported"
        )
    if (
        style.arrow_tip is not None
        or style.arrow_length_pt is not None
        or style.arrow_width_pt is not None
    ):
        raise PlanarCurveStyleError(
            "explicit 3D planar curve v1 does not support arrow tips"
        )
    if isinstance(style.rotate_degrees, bool):
        raise PlanarCurveStyleError(
            "explicit 3D planar curve rotation must be a finite number"
        )
    try:
        rotation = float(style.rotate_degrees)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanarCurveStyleError(
            "explicit 3D planar curve rotation must be a finite number"
        ) from exc
    if (
        style.transform_shape
        or style.native_canvas_plane is not None
        or not isfinite(rotation)
        or rotation != 0.0
    ):
        raise PlanarCurveStyleError(
            "explicit 3D planar curves do not accept an additional canvas transform"
        )
    if (
        style.font_command is not None
        or style.inner_xsep_pt is not None
        or style.inner_ysep_pt is not None
        or style.text_color is not None
        or style.node_border_color is not None
        or style.rectangle_node
    ):
        raise PlanarCurveStyleError(
            "node-only style fields are invalid for an explicit 3D planar curve"
        )
    if isinstance(style.line_width_pt, bool):
        raise PlanarCurveStyleError(
            "explicit 3D planar curve line width must be finite and positive"
        )
    try:
        width = float(style.line_width_pt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanarCurveStyleError(
            "explicit 3D planar curve line width must be finite and positive"
        ) from exc
    if not isfinite(width) or width <= 0.0:
        raise PlanarCurveStyleError(
            "explicit 3D planar curve line width must be finite and positive"
        )
    if style.line_cap not in {"round", "butt", "square"}:
        raise PlanarCurveStyleError(
            "explicit 3D planar curve line cap is unsupported"
        )
    if style.line_join not in {"round", "bevel", "miter"}:
        raise PlanarCurveStyleError(
            "explicit 3D planar curve line join is unsupported"
        )
    raw_values = [style.opacity]
    if style.draw_opacity is not None:
        raw_values.append(style.draw_opacity)
    if any(isinstance(value, bool) for value in raw_values):
        raise PlanarCurveStyleError(
            "explicit 3D planar curve opacity must lie between zero and one"
        )
    try:
        values = [float(value) for value in raw_values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanarCurveStyleError(
            "explicit 3D planar curve opacity must lie between zero and one"
        ) from exc
    if any(not isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise PlanarCurveStyleError(
            "explicit 3D planar curve opacity must lie between zero and one"
        )
    effective = values[0] * (1.0 if len(values) == 1 else values[1])
    if not isfinite(effective) or effective <= 0.0:
        raise PlanarCurveStyleError(
            "explicit 3D planar curves require a visible stroke opacity"
        )


__all__ = [
    "PlanarCurveStyleError",
    "certify_planar_curve_affine_display",
    "certify_planar_curve_display_segment",
    "certify_planar_curve_display_scale",
    "validate_planar_curve_stroke_style",
]
