from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from manim import (
    Animation,
    AnimationGroup,
    Create,
    FadeIn,
    GrowFromCenter,
    LaggedStart,
    Mobject,
    Scene,
    VGroup,
    Write,
)

from .compiler import ObjectSpec, PictureSpec
from .manim_renderer import NativeFigure


LABEL_KINDS = frozenset({"label", "path_label", "angle_label"})
MARKER_KINDS = frozenset({"angle", "right_angle"})

SEMANTIC_LAYER_ORDER = (
    "fills",
    "coordinate_frame",
    "solid_geometry",
    "auxiliary_geometry",
    "markers",
    "points",
    "labels",
)

DEFAULT_LAYER_RUN_TIMES = {
    "fills": 0.55,
    "coordinate_frame": 1.25,
    "solid_geometry": 1.35,
    "auxiliary_geometry": 0.85,
    "markers": 0.60,
    "points": 0.60,
    "labels": 1.45,
}


@dataclass(frozen=True)
class SemanticAnimationLayer:
    """Stable object IDs belonging to one automatic reveal phase."""

    name: str
    object_ids: tuple[str, ...]


def semantic_layer_name(spec: ObjectSpec) -> str:
    """Map every supported native object to one exhaustive animation layer."""

    if spec.kind == "polygon":
        return "fills"
    if spec.kind in {
        "arrow",
        "ellipse",
        "circle",
        "planar_circle_3d",
        "planar_ellipse_3d",
    }:
        return "coordinate_frame"
    if spec.kind == "line":
        return (
            "auxiliary_geometry"
            if spec.style.dash_pattern_pt
            else "solid_geometry"
        )
    if spec.kind == "dandelin_diagram":
        return "solid_geometry"
    if spec.kind in MARKER_KINDS:
        return "markers"
    if spec.kind == "dot":
        return "points"
    if spec.kind in LABEL_KINDS:
        return "labels"
    raise ValueError(f"No animation layer for native kind {spec.kind!r}")


def semantic_animation_layers(
    picture: PictureSpec,
    *,
    include_empty: bool = False,
) -> tuple[SemanticAnimationLayer, ...]:
    """Return deterministic layers while preserving source order inside each layer."""

    buckets = {name: [] for name in SEMANTIC_LAYER_ORDER}
    for spec in picture.objects:
        buckets[semantic_layer_name(spec)].append(spec.id)
    return tuple(
        SemanticAnimationLayer(name, tuple(buckets[name]))
        for name in SEMANTIC_LAYER_ORDER
        if include_empty or buckets[name]
    )


def _dashed_line_animation(mobject: Mobject) -> Animation:
    """Draw a native dashed line in path order instead of fading the group in."""

    if not isinstance(mobject, VGroup) or len(mobject) == 0:
        return Create(mobject)
    return AnimationGroup(
        *(Create(dash) for dash in mobject),
        lag_ratio=0.08,
    )


def native_reveal_animation(
    spec: ObjectSpec,
    mobject: Mobject,
    *,
    label_mode: str = "write",
) -> Animation:
    """Build the appropriate animation without flattening semantic objects."""

    if spec.kind == "polygon":
        return FadeIn(mobject)
    if spec.kind == "dot":
        return GrowFromCenter(mobject)
    if spec.kind in LABEL_KINDS:
        if label_mode == "write":
            return Write(mobject)
        if label_mode == "fade":
            return FadeIn(mobject)
        raise ValueError(f"Unknown label animation mode {label_mode!r}")
    if spec.kind == "line" and spec.style.dash_pattern_pt:
        return _dashed_line_animation(mobject)
    return Create(mobject)


def _specs_by_id(figure: NativeFigure) -> dict[str, ObjectSpec]:
    return {spec.id: spec for spec in figure.picture.objects}


def named_reveal_animation(
    figure: NativeFigure,
    object_ids: Sequence[str],
    *,
    lag_ratio: float = 0.08,
    label_mode: str = "write",
) -> Animation:
    """Build one staggered animation from explicit stable semantic IDs."""

    specs = _specs_by_id(figure)
    missing = [object_id for object_id in object_ids if object_id not in specs]
    if missing:
        raise KeyError(
            f"Picture {figure.picture.index} has no animation objects: {missing}"
        )
    if not object_ids:
        raise ValueError("A named reveal needs at least one object ID")
    animations = [
        native_reveal_animation(
            specs[object_id],
            figure.objects[object_id],
            label_mode=label_mode,
        )
        for object_id in object_ids
    ]
    return LaggedStart(*animations, lag_ratio=lag_ratio)


def play_named_reveal(
    scene: Scene,
    figure: NativeFigure,
    object_ids: Sequence[str],
    *,
    run_time: float,
    lag_ratio: float = 0.08,
    label_mode: str = "write",
) -> None:
    """Reveal selected IDs while keeping every line, point and label independent."""

    scene.play(
        named_reveal_animation(
            figure,
            object_ids,
            lag_ratio=lag_ratio,
            label_mode=label_mode,
        ),
        run_time=run_time,
    )


def play_semantic_reveal(
    scene: Scene,
    figures: Iterable[NativeFigure],
    *,
    layer_run_times: Mapping[str, float] | None = None,
    object_lag_ratio: float = 0.035,
    label_mode: str = "write",
) -> None:
    """Automatically reveal one or many figures in deterministic semantic layers."""

    native_figures = tuple(figures)
    run_times = dict(DEFAULT_LAYER_RUN_TIMES)
    if layer_run_times:
        run_times.update(layer_run_times)
    specs_by_figure = {
        id(figure): _specs_by_id(figure) for figure in native_figures
    }

    for layer_name in SEMANTIC_LAYER_ORDER:
        animations: list[Animation] = []
        for figure in native_figures:
            specs = specs_by_figure[id(figure)]
            layer_ids = next(
                (
                    layer.object_ids
                    for layer in semantic_animation_layers(figure.picture)
                    if layer.name == layer_name
                ),
                (),
            )
            animations.extend(
                native_reveal_animation(
                    specs[object_id],
                    figure.objects[object_id],
                    label_mode=label_mode,
                )
                for object_id in layer_ids
            )
        if animations:
            scene.play(
                LaggedStart(*animations, lag_ratio=object_lag_ratio),
                run_time=run_times[layer_name],
            )
