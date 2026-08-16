"""Manim binding adapter for globally solved TikZ Native 3D visibility.

The pure TikZ adapter proves topology and produces a deterministic visibility
model.  This module performs the separate, reversible task of binding that
model to one already-instantiated ``NativeFigure``.  It deliberately does not
modify the legacy compiler, renderer, Geometry Rig, or v1/v2 Native sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ContextManager, Literal, Mapping, Sequence

import numpy as np
from manim import Line, Mobject

from polyhedron_visibility import (
    AutoOcclusion3D,
    OcclusionStyle,
    ParallelProjection,
    TolerancePolicy,
)
from polyhedron_visibility.binding import DisplayPointProvider
from polyhedron_visibility.trace import VisibilityFrame

from .compiler import PictureSpec
from .manim_renderer import NativeFigure
from .polyhedron_visibility_3d_adapter import (
    TikzNativeVisibility3DAdapterResult,
    adapt_picture_visibility_3d,
)


class TikzNativeVisibility3DManimError(ValueError):
    """Raised before Scene mutation when a TikZ binding is ambiguous."""


CoordinateProvider = Callable[[], Mapping[str, Sequence[float]]]


def _point3(value: Sequence[float], label: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise TikzNativeVisibility3DManimError(
            f"{label} must be a finite three-component point"
        )
    return point


def _canonical_position_provider(
    result: TikzNativeVisibility3DAdapterResult,
    provider: CoordinateProvider | None,
) -> CoordinateProvider:
    if provider is None:
        frozen = {
            key: np.asarray(value, dtype=float)
            for key, value in result.model.entry_positions.items()
        }
        return lambda: {key: value.copy() for key, value in frozen.items()}

    aliases: dict[str, list[str]] = {}
    for authored, canonical in result.coordinate_vertex_ids:
        aliases.setdefault(canonical, []).append(authored)
    tolerance_policy = TolerancePolicy()

    def current() -> dict[str, np.ndarray]:
        raw = provider()
        if not isinstance(raw, Mapping):
            raise TikzNativeVisibility3DManimError(
                "coordinate_provider must return a mapping"
            )
        relevant_names = {
            name
            for vertex in result.model.vertices
            for name in (vertex.vertex_id, *aliases.get(vertex.vertex_id, ()))
        }
        parsed = {
            name: _point3(raw[name], f"coordinate {name}")
            for name in sorted(relevant_names)
            if name in raw
        }
        if not parsed:
            raise TikzNativeVisibility3DManimError(
                "coordinate_provider omitted all visibility coordinates"
            )
        alias_tolerance = tolerance_policy.resolve(tuple(parsed.values())).world
        positions: dict[str, np.ndarray] = {}
        for vertex in result.model.vertices:
            names = [vertex.vertex_id, *aliases.get(vertex.vertex_id, ())]
            values = [
                parsed[name]
                for name in dict.fromkeys(names)
                if name in parsed
            ]
            if not values:
                raise TikzNativeVisibility3DManimError(
                    f"coordinate_provider omitted {vertex.vertex_id}"
                )
            if any(
                float(np.linalg.norm(values[0] - value)) > alias_tolerance
                for value in values[1:]
            ):
                raise TikzNativeVisibility3DManimError(
                    f"welded aliases for {vertex.vertex_id} disagree"
                )
            positions[vertex.vertex_id] = values[0]
        return positions

    return current


def _single_object_stroke_bindings(
    result: TikzNativeVisibility3DAdapterResult,
    figure: NativeFigure,
) -> dict[str, Mobject]:
    bindings: dict[str, Mobject] = {}
    for stroke in result.stroke_bindings:
        if len(stroke.object_ids) != 1:
            raise TikzNativeVisibility3DManimError(
                "automatic Manim binding v1 requires one complete source Mobject per "
                f"semantic stroke; {stroke.source_edge_id} owns {len(stroke.object_ids)}"
            )
        object_id = stroke.object_ids[0]
        source = figure.objects.get(object_id)
        if source is None:
            raise TikzNativeVisibility3DManimError(
                f"NativeFigure omitted source object {object_id}"
            )
        drawable = [
            member
            for member in source.get_family()
            if np.asarray(getattr(member, "points", ()), dtype=float).size > 0
        ]
        if not isinstance(source, Line) or drawable != [source]:
            raise TikzNativeVisibility3DManimError(
                "automatic Manim binding v1 requires one continuous straight Line "
                f"per semantic stroke; {stroke.source_edge_id} is compound or dashed"
            )
        bindings[stroke.source_edge_id] = source
    return bindings


def _fit_entry_display_mapper(
    picture: PictureSpec,
    figure: NativeFigure,
    result: TikzNativeVisibility3DAdapterResult,
) -> Callable[[Sequence[float], Sequence[Sequence[float]]], np.ndarray]:
    """Fit the ShapeState's existing 2D affine placement from named lines."""

    object_specs = {item.id: item for item in picture.objects}
    coordinates = result.model.entry_positions
    alias_map = result.coordinate_vertex_map
    projection = np.asarray(result.entry_projection, dtype=float)
    logical_rows: list[tuple[float, float, float]] = []
    scene_rows: list[np.ndarray] = []
    for stroke in result.stroke_bindings:
        if len(stroke.object_ids) != 1:
            continue
        object_id = stroke.object_ids[0]
        spec = object_specs.get(object_id)
        mobject = figure.objects.get(object_id)
        if spec is None or mobject is None or spec.kind != "line":
            continue
        start_name = spec.geometry.get("start_name")
        end_name = spec.geometry.get("end_name")
        if not isinstance(start_name, str) or not isinstance(end_name, str):
            continue
        if start_name not in alias_map or end_name not in alias_map:
            continue
        logical_points = (
            projection @ np.asarray(coordinates[alias_map[start_name]], dtype=float),
            projection @ np.asarray(coordinates[alias_map[end_name]], dtype=float),
        )
        scene_points = (mobject.get_start(), mobject.get_end())
        for logical, scene_point in zip(logical_points, scene_points):
            logical_rows.append((float(logical[0]), float(logical[1]), 1.0))
            scene_rows.append(_point3(scene_point, f"source object {object_id}"))
    if len(logical_rows) < 3:
        raise TikzNativeVisibility3DManimError(
            "at least three named source endpoints are required to recover ShapeState placement"
        )
    logical_matrix = np.asarray(logical_rows, dtype=float)
    scene_matrix = np.asarray(scene_rows, dtype=float)
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(
        logical_matrix,
        scene_matrix,
        rcond=None,
    )
    if rank != 3:
        raise TikzNativeVisibility3DManimError(
            "named source endpoints do not determine a 2D ShapeState placement"
        )
    fitted = logical_matrix @ coefficients
    scene_scale = max(
        1.0,
        float(
            np.linalg.norm(
                np.max(scene_matrix, axis=0) - np.min(scene_matrix, axis=0)
            )
        ),
    )
    if float(np.max(np.linalg.norm(fitted - scene_matrix, axis=1))) > 1.0e-7 * scene_scale:
        raise TikzNativeVisibility3DManimError(
            "NativeFigure endpoints are not one stable affine placement of the visibility model"
        )

    def map_point(
        world: Sequence[float],
        matrix: Sequence[Sequence[float]],
    ) -> np.ndarray:
        projected = np.asarray(matrix, dtype=float) @ _point3(world, "world point")
        return np.asarray((projected[0], projected[1], 1.0)) @ coefficients

    return map_point


@dataclass(frozen=True)
class TikzNativeAutoOcclusion3D:
    """A proven TikZ contract plus its reversible Manim controller."""

    analysis: TikzNativeVisibility3DAdapterResult
    controller: AutoOcclusion3D

    @property
    def last_frame(self) -> VisibilityFrame | None:
        return self.controller.last_frame

    def attach(self) -> "TikzNativeAutoOcclusion3D":
        self.controller.attach()
        return self

    def update(self, dt: float = 0.0) -> "TikzNativeAutoOcclusion3D":
        self.controller.update(dt)
        return self

    def restore(self) -> "TikzNativeAutoOcclusion3D":
        self.controller.restore()
        return self

    def session(self) -> ContextManager[AutoOcclusion3D]:
        return self.controller.session()


def bind_picture_visibility_3d(
    scene: object,
    picture: PictureSpec,
    figure: NativeFigure,
    *,
    style: OcclusionStyle,
    validation_mode: Literal[
        "closed_convex_polyhedron", "independent_convex_faces"
    ] = "closed_convex_polyhedron",
    coordinate_provider: CoordinateProvider | None = None,
    projection: ParallelProjection | None = None,
    display_point_provider: DisplayPointProvider | None = None,
) -> TikzNativeAutoOcclusion3D:
    """Prove a TikZ face system and bind it to an existing NativeFigure.

    The default path is static-world geometry with the authored TikZ parallel
    projection.  Dynamic Geometry Rig callers can provide live coordinates and
    a live ``ParallelProjection``.  When no display mapper is supplied, the
    entry ShapeState affine placement is recovered from complete named lines
    and reused for every later projection matrix.
    """

    analysis = adapt_picture_visibility_3d(
        picture,
        validation_mode=validation_mode,
    )
    positions = _canonical_position_provider(analysis, coordinate_provider)
    stroke_bindings = _single_object_stroke_bindings(analysis, figure)
    current_projection = projection or ParallelProjection(analysis.entry_projection)
    if display_point_provider is None:
        mapper = _fit_entry_display_mapper(picture, figure, analysis)

        def fitted_display_point(world: Sequence[float]) -> Sequence[float]:
            return mapper(world, current_projection.current_matrix(scene))

        display_point_provider = fitted_display_point

    controller = AutoOcclusion3D(
        scene,
        analysis.model,
        position_provider=positions,
        stroke_bindings=stroke_bindings,
        projection=current_projection,
        display_point_provider=display_point_provider,
        style=style,
        require_closed_convex_manifold=(
            analysis.validation_mode == "closed_convex_polyhedron"
        ),
        source_coordinate_mode="display",
    )
    return TikzNativeAutoOcclusion3D(analysis, controller)


__all__ = [
    "CoordinateProvider",
    "TikzNativeAutoOcclusion3D",
    "TikzNativeVisibility3DManimError",
    "bind_picture_visibility_3d",
]
