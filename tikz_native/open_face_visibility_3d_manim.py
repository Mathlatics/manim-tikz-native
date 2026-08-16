"""Transactional Manim binding for proven TikZ open-face visibility.

Legacy relation fragments are never treated as dynamic geometry.  After the
adapter proves that they partition complete logical strokes, this binding uses
off-scene complete ``Line`` proxies to allocate stable Cairo overlay slots,
hides the authored fragments transactionally, and restores their exact vector
style on every normal or exceptional exit.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil, isfinite
import threading
from typing import Callable, ContextManager, Iterator, Mapping, Sequence
import weakref

import numpy as np
from manim import (
    CapStyleType,
    Line,
    LineJointType,
    Mobject,
    Polygon,
    ThreeDCamera,
    VGroup,
)

from polyhedron_visibility import OcclusionStyle, ParallelProjection, TolerancePolicy
from polyhedron_visibility.binding import (
    DisplayPointProvider,
    OverlayCapacity,
    OverlayPlan,
    _StrokeSlots,
    _capture_family_style,
    _drawable_member,
    _hide_snapshots,
    _restore_snapshots,
    _using_cairo_renderer,
    build_overlay_plan,
)
from polyhedron_visibility.open_faces import (
    OPEN_FACE_BINDING_SCALE_LIMITS,
    OpenFaceVisibilityFrame,
    compute_open_face_visibility,
)

from .compiler import ObjectSpec, PictureSpec, TEX_PT_PER_CM
from .manim_renderer import NativeFigure
from .open_face_visibility_3d_adapter import (
    LegacyRelationProof3D,
    TikzNativeOpenFaceVisibility3DAdapterResult,
    adapt_picture_open_face_visibility_3d,
)


CoordinateProvider = Callable[[], Mapping[str, Sequence[float]]]


class TikzNativeOpenFaceVisibility3DManimError(RuntimeError):
    """Raised before or during a fail-closed open-face binding operation."""


_FIGURE_OWNER_LOCK = threading.RLock()
_FIGURE_OWNERS: dict[int, weakref.ReferenceType["OpenFaceManimBinding3D"]] = {}
# Frozen NativeManim3DRenderer._native_dashes emits no child when a fragment's
# scene-space length is at or below this loop threshold.  The adapter's
# compiler proof remains usable at that extreme scale, so the new overlay can
# reconstruct the complete logical stroke without treating an empty authored
# dash group as arbitrary missing data.
_NATIVE_DASH_EMPTY_THRESHOLD = 1.0e-9


def _claim_figure_owner(controller: "OpenFaceManimBinding3D") -> None:
    key = id(controller.figure.group)
    with _FIGURE_OWNER_LOCK:
        reference = _FIGURE_OWNERS.get(key)
        owner = None if reference is None else reference()
        if owner is not None and owner is not controller:
            raise TikzNativeOpenFaceVisibility3DManimError(
                "NativeFigure already has an attached open-face visibility binding"
            )

        def release_dead(
            dead: weakref.ReferenceType["OpenFaceManimBinding3D"],
            *,
            identity: int = key,
        ) -> None:
            with _FIGURE_OWNER_LOCK:
                if _FIGURE_OWNERS.get(identity) is dead:
                    _FIGURE_OWNERS.pop(identity, None)

        _FIGURE_OWNERS[key] = weakref.ref(controller, release_dead)
        controller._owner_claimed = True


def _release_figure_owner(controller: "OpenFaceManimBinding3D") -> None:
    key = id(controller.figure.group)
    with _FIGURE_OWNER_LOCK:
        reference = _FIGURE_OWNERS.get(key)
        if reference is not None and reference() is controller:
            _FIGURE_OWNERS.pop(key, None)
        controller._owner_claimed = False


def _point3(value: object, label: str) -> np.ndarray:
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TikzNativeOpenFaceVisibility3DManimError(
            f"{label} must be numeric"
        ) from exc
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise TikzNativeOpenFaceVisibility3DManimError(
            f"{label} must be a finite three-component point"
        )
    return point


def _canonical_position_provider(
    result: TikzNativeOpenFaceVisibility3DAdapterResult,
    provider: CoordinateProvider | None,
    *,
    tolerance_policy: TolerancePolicy,
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

    def current() -> dict[str, np.ndarray]:
        raw = provider()
        if not isinstance(raw, Mapping):
            raise TikzNativeOpenFaceVisibility3DManimError(
                "coordinate_provider must return a mapping"
            )
        relevant = {
            name
            for vertex in result.model.vertices
            for name in (vertex.vertex_id, *aliases.get(vertex.vertex_id, ()))
        }
        parsed = {
            name: _point3(raw[name], f"coordinate {name}")
            for name in sorted(relevant)
            if name in raw
        }
        if not parsed:
            raise TikzNativeOpenFaceVisibility3DManimError(
                "coordinate_provider omitted all open-face coordinates"
            )
        alias_tolerance = tolerance_policy.resolve(tuple(parsed.values())).world
        positions: dict[str, np.ndarray] = {}
        for vertex in result.model.vertices:
            names = list(dict.fromkeys((vertex.vertex_id, *aliases.get(vertex.vertex_id, ()))))
            values = [parsed[name] for name in names if name in parsed]
            if not values:
                raise TikzNativeOpenFaceVisibility3DManimError(
                    f"coordinate_provider omitted {vertex.vertex_id}"
                )
            if any(
                float(np.linalg.norm(values[0] - value)) > alias_tolerance
                for value in values[1:]
            ):
                raise TikzNativeOpenFaceVisibility3DManimError(
                    f"welded aliases for {vertex.vertex_id} disagree"
                )
            positions[vertex.vertex_id] = values[0]
        return positions

    return current


@dataclass(frozen=True)
class _EntryAffineMapper:
    coefficients: np.ndarray
    scene_units_per_coordinate_cm: float
    scene_units_per_tex_pt: float
    residual_tolerance: float

    def map_point(
        self,
        world: Sequence[float],
        projection_matrix: Sequence[Sequence[float]],
    ) -> np.ndarray:
        projected = np.asarray(projection_matrix, dtype=float) @ _point3(
            world, "world point"
        )
        return np.asarray((projected[0], projected[1], 1.0)) @ self.coefficients


def _fit_entry_display_mapper(
    picture: PictureSpec,
    figure: NativeFigure,
    result: TikzNativeOpenFaceVisibility3DAdapterResult,
    *,
    tolerance_policy: TolerancePolicy,
) -> _EntryAffineMapper:
    """Recover the instantiated ShapeState placement from proven face vertices."""

    specs = {item.id: item for item in picture.objects}
    alias_map = result.coordinate_vertex_map
    positions = result.model.entry_positions
    projection = np.asarray(result.entry_projection, dtype=float)
    logical_rows: list[tuple[float, float, float]] = []
    scene_rows: list[np.ndarray] = []
    for face in result.face_bindings:
        for object_id in face.object_ids:
            spec = specs.get(object_id)
            mobject = figure.objects.get(object_id)
            if spec is None or spec.kind != "polygon" or not isinstance(mobject, Polygon):
                raise TikzNativeOpenFaceVisibility3DManimError(
                    f"face object {object_id} must remain one native Polygon"
                )
            raw_names = spec.geometry.get("point_names")
            vertices = np.asarray(mobject.get_vertices(), dtype=float)
            if (
                not isinstance(raw_names, (tuple, list))
                or len(raw_names) != len(vertices)
                or len(raw_names) < 3
            ):
                raise TikzNativeOpenFaceVisibility3DManimError(
                    f"face object {object_id} lost its named vertex correspondence"
                )
            for name, scene_point in zip(raw_names, vertices):
                if not isinstance(name, str) or name not in alias_map:
                    raise TikzNativeOpenFaceVisibility3DManimError(
                        f"face object {object_id} references an unknown coordinate"
                    )
                logical = projection @ np.asarray(positions[alias_map[name]], dtype=float)
                logical_rows.append((float(logical[0]), float(logical[1]), 1.0))
                scene_rows.append(_point3(scene_point, f"face object {object_id} vertex"))
    if len(logical_rows) < 3:
        raise TikzNativeOpenFaceVisibility3DManimError(
            "at least three face vertices are required to recover ShapeState placement"
        )
    logical_matrix = np.asarray(logical_rows, dtype=float)
    scene_matrix = np.asarray(scene_rows, dtype=float)
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(
        logical_matrix, scene_matrix, rcond=None
    )
    if rank != 3:
        raise TikzNativeOpenFaceVisibility3DManimError(
            "face vertices do not determine one 2D ShapeState placement"
        )
    tolerance = tolerance_policy.resolve(scene_matrix).boundary
    fitted = logical_matrix @ coefficients
    if float(np.max(np.linalg.norm(fitted - scene_matrix, axis=1))) > tolerance:
        raise TikzNativeOpenFaceVisibility3DManimError(
            "face polygons are not one stable affine placement of the open-face model"
        )
    first_axis = coefficients[0]
    second_axis = coefficients[1]
    first_length = float(np.linalg.norm(first_axis))
    second_length = float(np.linalg.norm(second_axis))
    scale = 0.5 * (first_length + second_length)
    if (
        scale <= tolerance_policy.absolute_floor
        or abs(first_length - second_length) > tolerance
        or abs(float(np.dot(first_axis, second_axis)))
        > tolerance_policy.angular * max(first_length * second_length, tolerance**2)
    ):
        raise TikzNativeOpenFaceVisibility3DManimError(
            "non-uniform or skewed ShapeState placement cannot preserve TikZ dash lengths"
        )
    picture_scale = float(picture.scale)
    if not isfinite(picture_scale) or picture_scale <= 0.0:
        raise TikzNativeOpenFaceVisibility3DManimError(
            "picture.scale must be finite and positive"
        )
    return _EntryAffineMapper(
        coefficients,
        scale,
        scale / (picture_scale * TEX_PT_PER_CM),
        tolerance,
    )


def _drawable_lines(source: Mobject, label: str) -> list[Line]:
    # Geometry Rig deliberately sets compiler fragments to opacity zero before
    # the global binding attaches.  Geometry proof therefore uses point-bearing
    # members, not current visibility; z-order conflict checks remain based on
    # genuinely visible drawables through ``_drawable_member``.
    drawable = [
        member
        for member in source.get_family()
        if np.asarray(getattr(member, "points", ()), dtype=float).size > 0
    ]
    if not drawable or any(not isinstance(member, Line) for member in drawable):
        raise TikzNativeOpenFaceVisibility3DManimError(
            f"{label} must contain only straight native Line drawables"
        )
    return list(drawable)  # type: ignore[return-value]


def _line_is_straight(line: Line, tolerance: float) -> bool:
    start = _point3(line.get_start(), "line start")
    end = _point3(line.get_end(), "line end")
    chord = end - start
    length = float(np.linalg.norm(chord))
    points = np.asarray(line.points, dtype=float)
    if length <= tolerance or points.ndim != 2 or points.shape[1:] != (3,):
        return False
    distances = np.linalg.norm(np.cross(points - start, chord), axis=1) / length
    return float(np.max(distances, initial=0.0)) <= tolerance


def _line_endpoint_parameters(
    line: Line,
    full_start: np.ndarray,
    full_end: np.ndarray,
) -> tuple[float, float]:
    delta = full_end - full_start
    denominator = float(np.dot(delta, delta))
    return tuple(
        float(np.dot(_point3(point, "line endpoint") - full_start, delta) / denominator)
        for point in (line.get_start(), line.get_end())
    )  # type: ignore[return-value]


def _line_distance_from_axis(
    line: Line,
    full_start: np.ndarray,
    full_end: np.ndarray,
) -> float:
    delta = full_end - full_start
    length = float(np.linalg.norm(delta))
    if length <= 0.0:
        return float("inf")
    points = np.asarray(
        [line.get_start(), line.get_end(), *np.asarray(line.points, dtype=float)],
        dtype=float,
    )
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.all(np.isfinite(points)):
        return float("inf")
    distances = np.linalg.norm(np.cross(points - full_start, delta), axis=1) / length
    return float(np.max(distances, initial=0.0))


def _validate_runtime_sources(
    picture: PictureSpec,
    figure: NativeFigure,
    result: TikzNativeOpenFaceVisibility3DAdapterResult,
    mapper: _EntryAffineMapper,
    *,
    tolerance_policy: TolerancePolicy,
) -> None:
    specs = {item.id: item for item in picture.objects}
    direct_children = {id(item) for item in figure.group.submobjects}
    projection = result.entry_projection
    entry_positions = result.model.entry_positions
    proofs_by_edge: dict[str, list[LegacyRelationProof3D]] = {}
    for proof in result.relation_proofs:
        proofs_by_edge.setdefault(proof.source_edge_id, []).append(proof)

    for binding in result.stroke_bindings:
        proofs = proofs_by_edge.get(binding.source_edge_id, [])
        if not proofs:
            if len(binding.object_ids) != 1:
                raise TikzNativeOpenFaceVisibility3DManimError(
                    f"plain stroke {binding.source_edge_id} must own one complete Line"
                )
            object_id = binding.object_ids[0]
            source = figure.objects.get(object_id)
            spec = specs.get(object_id)
            if (
                source is None
                or spec is None
                or id(source) not in direct_children
                or not isinstance(source, Line)
                or _drawable_lines(source, object_id) != [source]
                or spec.style.dash_pattern_pt is not None
                or spec.style.arrow_tip is not None
            ):
                raise TikzNativeOpenFaceVisibility3DManimError(
                    f"plain stroke {binding.source_edge_id} must remain one solid native Line"
                )
            oriented = result.model.stroke_map[binding.source_edge_id].vertex_ids
            expected_start = mapper.map_point(entry_positions[oriented[0]], projection)
            expected_end = mapper.map_point(entry_positions[oriented[1]], projection)
            full_length = float(np.linalg.norm(expected_end - expected_start))
            tolerance = max(
                mapper.residual_tolerance,
                tolerance_policy.resolve(
                    (expected_start, expected_end), edge_length=full_length
                ).boundary,
            )
            if not _line_is_straight(source, tolerance):
                raise TikzNativeOpenFaceVisibility3DManimError(
                    f"plain stroke {binding.source_edge_id} is not straight"
                )
            actual = (source.get_start(), source.get_end())
            forward = all(
                np.linalg.norm(_point3(value, object_id) - expected) <= tolerance
                for value, expected in zip(actual, (expected_start, expected_end))
            )
            reverse = all(
                np.linalg.norm(_point3(value, object_id) - expected) <= tolerance
                for value, expected in zip(actual, (expected_end, expected_start))
            )
            if not (forward or reverse):
                raise TikzNativeOpenFaceVisibility3DManimError(
                    f"plain stroke {binding.source_edge_id} no longer matches its full segment"
                )
            continue

        for proof in proofs:
            full_start = mapper.map_point(
                entry_positions[proof.canonical_vertex_ids[0]], projection
            )
            full_end = mapper.map_point(
                entry_positions[proof.canonical_vertex_ids[1]], projection
            )
            full_delta = full_end - full_start
            full_length = float(np.linalg.norm(full_delta))
            resolved = tolerance_policy.resolve(
                (full_start, full_end), edge_length=full_length
            )
            tolerance = max(mapper.residual_tolerance, resolved.boundary)
            parameter_tolerance = max(
                resolved.parameter,
                tolerance / max(full_length, tolerance_policy.absolute_floor),
            )
            for fragment in proof.fragments:
                source = figure.objects.get(fragment.object_id)
                spec = specs.get(fragment.object_id)
                if source is None or spec is None or id(source) not in direct_children:
                    raise TikzNativeOpenFaceVisibility3DManimError(
                        f"proven fragment {fragment.object_id} is missing or no longer a direct child"
                    )
                dashed = spec.style.dash_pattern_pt is not None
                point_members = [
                    member
                    for member in source.get_family()
                    if np.asarray(getattr(member, "points", ()), dtype=float).size > 0
                ]
                expected_fragment_length = (
                    fragment.end_parameter - fragment.start_parameter
                ) * full_length
                if (
                    dashed
                    and not point_members
                    and expected_fragment_length
                    <= _NATIVE_DASH_EMPTY_THRESHOLD + tolerance
                ):
                    continue
                lines = _drawable_lines(source, f"fragment {fragment.object_id}")
                if dashed:
                    if isinstance(source, Line):
                        raise TikzNativeOpenFaceVisibility3DManimError(
                            f"dashed fragment {fragment.object_id} lost its native dash group"
                        )
                elif not isinstance(source, Line) or lines != [source]:
                    raise TikzNativeOpenFaceVisibility3DManimError(
                        f"solid fragment {fragment.object_id} must remain one native Line"
                    )
                parameters: list[tuple[float, float]] = []
                for line in lines:
                    if not _line_is_straight(line, tolerance):
                        raise TikzNativeOpenFaceVisibility3DManimError(
                            f"fragment {fragment.object_id} contains a curved drawable"
                        )
                    if _line_distance_from_axis(line, full_start, full_end) > tolerance:
                        raise TikzNativeOpenFaceVisibility3DManimError(
                            f"fragment {fragment.object_id} is offset from its complete logical Line"
                        )
                    start_t, end_t = _line_endpoint_parameters(line, full_start, full_end)
                    low, high = sorted((start_t, end_t))
                    if (
                        low < fragment.start_parameter - parameter_tolerance
                        or high > fragment.end_parameter + parameter_tolerance
                    ):
                        raise TikzNativeOpenFaceVisibility3DManimError(
                            f"fragment {fragment.object_id} escapes its proven parameter interval"
                        )
                    parameters.append((low, high))
                parameters.sort()
                if dashed:
                    if (
                        abs(parameters[0][0] - fragment.start_parameter)
                        > parameter_tolerance
                    ):
                        raise TikzNativeOpenFaceVisibility3DManimError(
                            f"dashed fragment {fragment.object_id} lost its source-edge dash phase"
                        )
                else:
                    low, high = parameters[0]
                    if (
                        abs(low - fragment.start_parameter) > parameter_tolerance
                        or abs(high - fragment.end_parameter) > parameter_tolerance
                    ):
                        raise TikzNativeOpenFaceVisibility3DManimError(
                            f"solid fragment {fragment.object_id} no longer covers its proven interval"
                        )


def _scene_containers(scene: object) -> tuple[list[object], ...]:
    result: list[list[object]] = []
    for name in ("mobjects", "foreground_mobjects", "moving_mobjects", "static_mobjects"):
        value = getattr(scene, name, None)
        if isinstance(value, list) and all(value is not item for item in result):
            result.append(value)
    return tuple(result)


def _scene_family(scene: object) -> list[object]:
    result: list[object] = []
    seen: set[int] = set()
    for container in _scene_containers(scene):
        for root in container:
            for member in root.get_family():
                if id(member) not in seen:
                    seen.add(id(member))
                    result.append(member)
    return result


def _managed_sources(
    figure: NativeFigure,
    result: TikzNativeOpenFaceVisibility3DAdapterResult,
) -> dict[str, Mobject]:
    sources: dict[str, Mobject] = {}
    for object_id in result.suppressed_object_ids:
        source = figure.objects.get(object_id)
        if source is None:
            raise TikzNativeOpenFaceVisibility3DManimError(
                f"NativeFigure omitted managed source {object_id}"
            )
        sources[object_id] = source
    return sources


def _geometry_rig_relation_sources(
    picture: PictureSpec,
    figure: NativeFigure,
    state: Mapping[str, object] | None,
) -> dict[str, Mobject]:
    """Validate an explicitly supplied native v2 Geometry Rig state.

    The generated rig replaces legacy compiler fragments with one temporary
    relation group per compiler relation.  These groups may only be taken over
    through the explicit state object returned by
    ``install_geometry_3d_updaters``; arbitrary VGroups are never inferred.
    """

    if state is None:
        return {}
    if state.get("shape") is not figure.group or state.get("objects") is not figure.objects:
        raise TikzNativeOpenFaceVisibility3DManimError(
            "geometry_rig_state does not own this NativeFigure"
        )
    original_children = state.get("original_shape_children")
    if not isinstance(original_children, list) or tuple(original_children) != tuple(
        figure.objects.values()
    ):
        raise TikzNativeOpenFaceVisibility3DManimError(
            "geometry_rig_state original ShapeState identity is incomplete"
        )
    entry_snapshots = state.get("entry_snapshots")
    if not isinstance(entry_snapshots, Mapping) or set(entry_snapshots) != set(
        figure.objects
    ):
        raise TikzNativeOpenFaceVisibility3DManimError(
            "geometry_rig_state entry snapshot identity is incomplete"
        )
    groups = state.get("temporary_groups")
    if not isinstance(groups, list) or len(groups) != len(picture.occlusion_relations):
        raise TikzNativeOpenFaceVisibility3DManimError(
            "geometry_rig_state must expose one temporary group per legacy relation"
        )
    direct = {id(item) for item in figure.group.submobjects}
    result: dict[str, Mobject] = {}
    for relation, raw_group in zip(picture.occlusion_relations, groups):
        if (
            not isinstance(raw_group, VGroup)
            or id(raw_group) not in direct
            or not raw_group.updaters
        ):
            raise TikzNativeOpenFaceVisibility3DManimError(
                f"geometry_rig_state relation {relation.id} has no active native VGroup"
            )
        lines = _drawable_lines(raw_group, f"geometry rig relation {relation.id}")
        if any(float(line.z_index) != float(relation.z_index) for line in lines):
            raise TikzNativeOpenFaceVisibility3DManimError(
                f"geometry rig relation {relation.id} has an unexpected z_index"
            )
        result[f"geometry-rig-relation:{relation.id}"] = raw_group
    return result


def _geometry_rig_is_restored(state: Mapping[str, object] | None) -> bool:
    if state is None:
        return False
    shape = state.get("shape")
    original = state.get("original_shape_children")
    groups = state.get("temporary_groups")
    if (
        not isinstance(shape, Mobject)
        or not isinstance(original, list)
        or not isinstance(groups, list)
    ):
        return False
    children = tuple(shape.submobjects)
    return len(children) == len(original) and all(
        child is entry for child, entry in zip(children, original)
    ) and all(
        child is not group for child in children for group in groups
    )


def _geometry_rig_project_scene(
    state: Mapping[str, object],
) -> Callable[[Sequence[float]], np.ndarray]:
    raw = state.get("project_scene")
    if not callable(raw):
        raise TikzNativeOpenFaceVisibility3DManimError(
            "geometry_rig_state has no live project_scene provider"
        )

    def project(point: Sequence[float]) -> np.ndarray:
        return _point3(raw(point), "geometry rig display point")

    return project


def _geometry_rig_projection_matrix(
    project_scene: Callable[[Sequence[float]], np.ndarray],
    mapper: _EntryAffineMapper,
    tolerance_policy: TolerancePolicy,
) -> tuple[tuple[float, float, float], ...]:
    """Recover the current local camera, removing ShapeState placement."""

    origin = project_scene((0.0, 0.0, 0.0))
    basis = tuple(
        project_scene(axis) - origin
        for axis in (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    scene_rows = np.asarray(
        (
            tuple(float(value[0]) for value in basis),
            tuple(float(value[1]) for value in basis),
        ),
        dtype=float,
    )
    placement = np.asarray(mapper.coefficients[:2, :2], dtype=float).T
    placement_scale = float(np.linalg.norm(placement, ord=2))
    placement_tolerance = tolerance_policy.resolve(
        (origin, *(origin + value for value in basis))
    ).world
    if (
        placement.shape != (2, 2)
        or not np.all(np.isfinite(placement))
        or abs(float(np.linalg.det(placement)))
        <= placement_tolerance * max(placement_scale, placement_tolerance)
    ):
        raise TikzNativeOpenFaceVisibility3DManimError(
            "ShapeState placement cannot recover the Geometry Rig camera"
        )
    screen_rows = np.linalg.solve(placement, scene_rows)
    depth = np.cross(screen_rows[0], screen_rows[1])
    depth_length = float(np.linalg.norm(depth))
    row_scale = max(
        float(np.linalg.norm(screen_rows[0]) * np.linalg.norm(screen_rows[1])),
        tolerance_policy.absolute_floor,
    )
    if depth_length <= tolerance_policy.angular * row_scale:
        raise TikzNativeOpenFaceVisibility3DManimError(
            "Geometry Rig local camera screen axes are singular"
        )
    depth /= depth_length
    matrix = np.vstack((screen_rows, depth))
    if not np.all(np.isfinite(matrix)) or float(np.linalg.det(matrix)) <= 0.0:
        raise TikzNativeOpenFaceVisibility3DManimError(
            "Geometry Rig local camera is not finite and right-handed"
        )

    # project_scene must be affine.  This extra point catches wrappers whose
    # basis samples look valid but whose actual display mapping is nonlinear.
    probe = np.asarray((0.375, -0.25, 0.625), dtype=float)
    actual = project_scene(probe)
    predicted_xy = origin[:2] + scene_rows @ probe
    affine_tolerance = tolerance_policy.resolve((origin, actual)).boundary
    if float(np.linalg.norm(actual[:2] - predicted_xy)) > affine_tolerance:
        raise TikzNativeOpenFaceVisibility3DManimError(
            "geometry_rig_state project_scene is not one affine parallel camera"
        )
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _validate_live_line_family(
    source: Mobject,
    full_start: np.ndarray,
    full_end: np.ndarray,
    *,
    label: str,
    tolerance_policy: TolerancePolicy,
    mapper_tolerance: float,
    require_complete_line: bool,
    active_only: bool,
    allow_empty: bool,
) -> None:
    """Prove that a current Manim source still represents its logical stroke.

    Legacy dashed groups may contain dormant zero-length pool members.  Those
    members carry no drawable interval and are ignored, while every
    non-degenerate member must remain collinear with and inside the complete
    logical segment.  A plain named stroke is stricter: it must remain exactly
    one complete Line, in either endpoint orientation.
    """

    delta = full_end - full_start
    full_length = float(np.linalg.norm(delta))
    resolved = tolerance_policy.resolve(
        (full_start, full_end), edge_length=full_length
    )
    tolerance = max(mapper_tolerance, resolved.boundary)
    parameter_tolerance = max(
        resolved.parameter,
        tolerance / max(full_length, tolerance_policy.absolute_floor),
    )
    point_members = [
        member
        for member in source.get_family()
        if np.asarray(getattr(member, "points", ()), dtype=float).size > 0
    ]
    if not point_members and allow_empty:
        return
    lines = _drawable_lines(source, label)
    if active_only:
        lines = [
            line
            for line in lines
            if (
                float(line.get_stroke_width()) > 0.0
                and float(line.get_stroke_opacity()) > 0.0
            )
            or (
                float(getattr(line, "background_stroke_width", 0.0)) > 0.0
                and float(getattr(line, "background_stroke_opacity", 0.0)) > 0.0
            )
        ]
        if not lines:
            # The binding itself suppresses the complete temporary group after
            # each successful frame.  A direct manual update before the outer
            # Geometry Rig updater runs therefore has no live fragment evidence
            # to re-check; coordinates + project_scene remain authoritative.
            # During normal Scene updates the rig runs first, active evidence is
            # present, and the collinearity proof below is enforced.
            return
    if full_length <= tolerance:
        # A valid 3D edge may collapse to one display point when viewed
        # exactly end-on.  The semantic 3D segment remains non-degenerate; the
        # runtime evidence is safe only if every current drawable collapses to
        # the same display point as well.
        for line in lines:
            points = np.asarray(
                [line.get_start(), line.get_end(), *np.asarray(line.points, dtype=float)],
                dtype=float,
            )
            if (
                points.ndim != 2
                or points.shape[1:] != (3,)
                or not np.all(np.isfinite(points))
                or float(np.max(np.linalg.norm(points - full_start, axis=1), initial=0.0))
                > tolerance
            ):
                raise TikzNativeOpenFaceVisibility3DManimError(
                    f"{label} does not collapse with its end-on logical Line"
                )
        return
    active: list[Line] = []
    for line in lines:
        start = _point3(line.get_start(), f"{label} start")
        end = _point3(line.get_end(), f"{label} end")
        line_length = float(np.linalg.norm(end - start))
        if line_length <= tolerance:
            continue
        active.append(line)
        if not _line_is_straight(line, tolerance):
            raise TikzNativeOpenFaceVisibility3DManimError(
                f"{label} contains a curved drawable"
            )
        if _line_distance_from_axis(line, full_start, full_end) > tolerance:
            raise TikzNativeOpenFaceVisibility3DManimError(
                f"{label} is offset from its current logical Line"
            )
        low, high = sorted(_line_endpoint_parameters(line, full_start, full_end))
        if low < -parameter_tolerance or high > 1.0 + parameter_tolerance:
            raise TikzNativeOpenFaceVisibility3DManimError(
                f"{label} escapes its current logical Line"
            )
    if not active:
        raise TikzNativeOpenFaceVisibility3DManimError(
            f"{label} has no non-degenerate Line evidence"
        )
    if require_complete_line:
        if len(lines) != 1 or len(active) != 1 or not isinstance(source, Line):
            raise TikzNativeOpenFaceVisibility3DManimError(
                f"{label} must remain one complete native Line"
            )
        actual = (_point3(source.get_start(), label), _point3(source.get_end(), label))
        forward = all(
            float(np.linalg.norm(value - expected)) <= tolerance
            for value, expected in zip(actual, (full_start, full_end))
        )
        reverse = all(
            float(np.linalg.norm(value - expected)) <= tolerance
            for value, expected in zip(actual, (full_end, full_start))
        )
        if not (forward or reverse):
            raise TikzNativeOpenFaceVisibility3DManimError(
                f"{label} no longer covers its complete logical Line"
            )


def _surrogate_z_indices(
    scene: object,
    picture: PictureSpec,
    figure: NativeFigure,
    result: TikzNativeOpenFaceVisibility3DAdapterResult,
    sources: Mapping[str, Mobject],
) -> dict[str, float]:
    family = _scene_family(scene)
    family_ids = {id(item) for item in family}
    if id(figure.group) not in family_ids:
        raise TikzNativeOpenFaceVisibility3DManimError(
            "NativeFigure.group is not owned by the current Scene"
        )
    managed_family_ids = {
        id(member) for source in sources.values() for member in source.get_family()
    }
    specs = {item.id: item for item in picture.objects}
    child_indices = {id(item): index for index, item in enumerate(figure.group.submobjects)}
    base_edges: dict[float, list[tuple[int, str]]] = {}

    for binding in result.stroke_bindings:
        base = float(binding.z_index)
        if not isfinite(base):
            raise TikzNativeOpenFaceVisibility3DManimError(
                f"stroke {binding.source_edge_id} has non-finite z_index"
            )
        source_indices: list[int] = []
        selected = False
        for object_id in binding.object_ids:
            source = sources[object_id]
            if id(source) not in child_indices:
                raise TikzNativeOpenFaceVisibility3DManimError(
                    f"managed source {object_id} is no longer a direct NativeFigure child"
                )
            source_indices.append(child_indices[id(source)])
            spec = specs.get(object_id)
            if spec is None:
                raise TikzNativeOpenFaceVisibility3DManimError(
                    f"managed source {object_id} has no ObjectSpec"
                )
            expected_z = float(spec.z_index)
            point_members = [
                member
                for member in source.get_family()
                if np.asarray(getattr(member, "points", ()), dtype=float).size > 0
            ]
            if not point_members and spec.style.dash_pattern_pt is not None:
                drawable: list[Line] = []
            else:
                drawable = _drawable_lines(source, object_id)
            if any(float(member.z_index) != expected_z for member in drawable):
                raise TikzNativeOpenFaceVisibility3DManimError(
                    f"managed source {object_id} no longer has its authored z_index"
                )
            selected = selected or expected_z == base
        if not selected:
            raise TikzNativeOpenFaceVisibility3DManimError(
                f"stroke {binding.source_edge_id} has no source at selected z_index {base}"
            )
        base_edges.setdefault(base, []).append((min(source_indices), binding.source_edge_id))

    drawable_scene = [member for member in family if _drawable_member(member)]
    bases = set(base_edges)
    for member in drawable_scene:
        value = float(member.z_index)
        if value in bases and id(member) not in managed_family_ids:
            raise TikzNativeOpenFaceVisibility3DManimError(
                f"unmanaged drawable shares managed z_index {value}"
            )
    actual_values = sorted(
        {float(member.z_index) for member in drawable_scene if isfinite(float(member.z_index))}
    )
    result_z: dict[str, float] = {}
    for base, values in sorted(base_edges.items()):
        upper = next((value for value in actual_values if value > base), base + 1.0)
        ordered = sorted(values, key=lambda item: (item[0], item[1]))
        gap = upper - base
        if gap <= np.spacing(base if base else 1.0) * (len(ordered) + 2):
            raise TikzNativeOpenFaceVisibility3DManimError(
                f"no representable surrogate z layer above {base}"
            )
        for rank, (_index, edge_id) in enumerate(ordered, start=1):
            value = base + gap * rank / (len(ordered) + 1)
            if not base < value < upper:
                raise TikzNativeOpenFaceVisibility3DManimError(
                    f"failed to allocate surrogate z layer for {edge_id}"
                )
            result_z[edge_id] = value
    return result_z


def _style_number(
    payload: Mapping[str, object], key: str, default: float, label: str
) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError) as exc:
        raise TikzNativeOpenFaceVisibility3DManimError(
            f"{label}.{key} must be numeric"
        ) from exc
    if not isfinite(value):
        raise TikzNativeOpenFaceVisibility3DManimError(
            f"{label}.{key} must be finite"
        )
    return value


def _draw_opacity(payload: Mapping[str, object], label: str) -> float:
    value = _style_number(payload, "opacity", 1.0, label) * _style_number(
        payload, "drawOpacity", 1.0, label
    )
    if value < 0.0 or value > 1.0:
        raise TikzNativeOpenFaceVisibility3DManimError(
            f"{label} draw opacity must lie in [0, 1]"
        )
    return value


def _cap_style(value: object) -> CapStyleType:
    return {
        "round": CapStyleType.ROUND,
        "butt": CapStyleType.BUTT,
        "square": CapStyleType.SQUARE,
    }.get(value, CapStyleType.AUTO)


def _joint_style(value: object) -> LineJointType:
    return {
        "round": LineJointType.ROUND,
        "bevel": LineJointType.BEVEL,
        "miter": LineJointType.MITER,
    }.get(value, LineJointType.AUTO)


def _stroke_width_per_pt(
    picture: PictureSpec,
    figure: NativeFigure,
    result: TikzNativeOpenFaceVisibility3DAdapterResult,
) -> float:
    specs = {item.id: item for item in picture.objects}
    ratios: list[float] = []
    for object_id in result.suppressed_object_ids:
        spec = specs[object_id]
        width_pt = float(spec.style.line_width_pt)
        if width_pt <= 0.0:
            continue
        point_members = [
            member
            for member in figure.objects[object_id].get_family()
            if np.asarray(getattr(member, "points", ()), dtype=float).size > 0
        ]
        if not point_members and spec.style.dash_pattern_pt is not None:
            continue
        for line in _drawable_lines(figure.objects[object_id], object_id):
            ratios.append(float(line.get_stroke_width()) / width_pt)
    if not ratios or any(not isfinite(value) or value <= 0.0 for value in ratios):
        raise TikzNativeOpenFaceVisibility3DManimError(
            "managed fragments do not expose one finite stroke-width scale"
        )
    reference = float(np.median(ratios))
    if any(abs(value - reference) > 1.0e-7 * max(reference, value) for value in ratios):
        raise TikzNativeOpenFaceVisibility3DManimError(
            "managed fragments do not share one native stroke-width scale"
        )
    return reference


def _edge_occlusion_style(
    binding: object,
    *,
    capacity_style: OcclusionStyle,
    mapper: _EntryAffineMapper,
) -> OcclusionStyle:
    visible = getattr(binding, "visible_style")
    hidden = getattr(binding, "hidden_style")
    edge_id = str(getattr(binding, "source_edge_id"))
    if not isinstance(visible, Mapping) or not isinstance(hidden, Mapping):
        raise TikzNativeOpenFaceVisibility3DManimError(
            f"stroke {edge_id} styles must be mappings"
        )
    if visible.get("dashPatternPt") is not None or visible.get("arrowTip") is not None:
        raise TikzNativeOpenFaceVisibility3DManimError(
            f"stroke {edge_id} visible style must be one solid line"
        )
    raw_dash = hidden.get("dashPatternPt")
    if not isinstance(raw_dash, (tuple, list)) or len(raw_dash) != 2:
        raise TikzNativeOpenFaceVisibility3DManimError(
            f"stroke {edge_id} hidden style must declare one dash/gap pair"
        )
    dash_on = _style_number({"value": raw_dash[0]}, "value", 0.0, edge_id)
    dash_off = _style_number({"value": raw_dash[1]}, "value", 0.0, edge_id)
    if dash_on <= 0.0 or dash_off < 0.0:
        raise TikzNativeOpenFaceVisibility3DManimError(
            f"stroke {edge_id} hidden dash pattern is invalid"
        )
    visible_width_pt = _style_number(visible, "lineWidthPt", 0.9, edge_id)
    hidden_width_pt = _style_number(hidden, "lineWidthPt", visible_width_pt, edge_id)
    visible_opacity = _draw_opacity(visible, edge_id)
    hidden_opacity = _draw_opacity(hidden, edge_id)
    if visible_width_pt <= 0.0 or hidden_width_pt <= 0.0 or visible_opacity <= 0.0:
        raise TikzNativeOpenFaceVisibility3DManimError(
            f"stroke {edge_id} needs a positive visible style and hidden width"
        )
    visible_color = visible.get("drawColor")
    hidden_color = hidden.get("drawColor")
    if not isinstance(visible_color, str) or not isinstance(hidden_color, str):
        raise TikzNativeOpenFaceVisibility3DManimError(
            f"stroke {edge_id} needs explicit visible and hidden colors"
        )
    return OcclusionStyle(
        max_projected_length=capacity_style.max_projected_length,
        dash_length=dash_on * mapper.scene_units_per_tex_pt,
        dash_gap=dash_off * mapper.scene_units_per_tex_pt,
        visible_color=visible_color,
        hidden_color=hidden_color,
        visible_width_scale=1.0,
        hidden_width_scale=hidden_width_pt / visible_width_pt,
        visible_opacity_scale=1.0,
        hidden_opacity_scale=hidden_opacity / visible_opacity,
    )


def _complete_proxy_line(
    binding: object,
    *,
    stroke_width_per_pt: float,
) -> Line:
    visible = getattr(binding, "visible_style")
    edge_id = str(getattr(binding, "source_edge_id"))
    assert isinstance(visible, Mapping)
    color = visible.get("drawColor")
    assert isinstance(color, str)
    width_pt = _style_number(visible, "lineWidthPt", 0.9, edge_id)
    opacity = _draw_opacity(visible, edge_id)
    proxy = Line(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        buff=0,
        stroke_color=color,
        stroke_width=width_pt * stroke_width_per_pt,
        stroke_opacity=opacity,
        cap_style=_cap_style(visible.get("lineCap")),
        joint_type=_joint_style(visible.get("lineJoin")),
    )
    proxy.set_stroke(opacity=0.0, width=0.0, background=True)
    return proxy


def _capacity_for(
    stroke: object,
    model: object,
    style: OcclusionStyle,
) -> OverlayCapacity:
    excluded = set(getattr(stroke, "excluded_occluder_face_ids"))
    incidents = set(getattr(stroke, "incident_face_ids"))
    candidate_count = sum(
        1
        for face in getattr(model, "faces")
        if face.occludes_strokes and face.face_id not in incidents | excluded
    )
    hidden_slots = candidate_count
    if getattr(stroke, "visibility_mode") == "always_hidden":
        hidden_slots = max(1, hidden_slots)
    return OverlayCapacity(
        visible_slots=candidate_count + 1,
        hidden_slots=hidden_slots,
        dash_slots_per_hidden=(
            int(ceil(style.max_projected_length / style.dash_period))
            + hidden_slots
            + 1
        ),
        max_projected_length=style.max_projected_length,
    )


def _guard_realtime_scale(
    model: object,
    capacities: Mapping[str, OverlayCapacity],
) -> None:
    """Apply the public v1 limits before allocating any Manim overlay object."""

    limits = OPEN_FACE_BINDING_SCALE_LIMITS
    fixed = (
        ("faces", len(getattr(model, "faces")), limits.max_faces),
        ("strokes", len(getattr(model, "strokes")), limits.max_strokes),
        ("seams", len(getattr(model, "seams")), limits.max_seams),
    )
    for label, value, maximum in fixed:
        if value > maximum:
            raise TikzNativeOpenFaceVisibility3DManimError(
                f"open-face realtime binding {label}={value} exceeds fixed v1 limit {maximum}"
            )
    candidate_pairs = 0
    for stroke in getattr(model, "strokes"):
        incidents = set(stroke.incident_face_ids)
        excluded = set(stroke.excluded_occluder_face_ids)
        candidate_pairs += sum(
            1
            for face in getattr(model, "faces")
            if face.occludes_strokes and face.face_id not in incidents | excluded
        )
    if candidate_pairs > limits.max_candidate_pairs:
        raise TikzNativeOpenFaceVisibility3DManimError(
            "open-face realtime binding candidate_pairs="
            f"{candidate_pairs} exceeds fixed v1 limit {limits.max_candidate_pairs}"
        )
    line_slots = sum(
        capacity.visible_slots
        + capacity.hidden_slots * capacity.dash_slots_per_hidden
        for capacity in capacities.values()
    )
    if line_slots > limits.max_overlay_line_slots:
        raise TikzNativeOpenFaceVisibility3DManimError(
            "open-face realtime binding overlay_line_slots="
            f"{line_slots} exceeds fixed v1 limit {limits.max_overlay_line_slots}"
        )


class OpenFaceManimBinding3D:
    """Independent, fixed-capacity Cairo binding for one proven open-face model."""

    def __init__(
        self,
        scene: object,
        picture: PictureSpec,
        figure: NativeFigure,
        analysis: TikzNativeOpenFaceVisibility3DAdapterResult,
        *,
        position_provider: CoordinateProvider,
        projection: ParallelProjection | None,
        display_point_provider: DisplayPointProvider | None,
        capacity_style: OcclusionStyle,
        tolerance_policy: TolerancePolicy,
        geometry_rig_state: Mapping[str, object] | None,
    ) -> None:
        self.scene = scene
        self.picture = picture
        self.figure = figure
        self.analysis = analysis
        self.model = analysis.model
        self.position_provider = position_provider
        self.tolerance_policy = tolerance_policy
        self._geometry_rig_state = geometry_rig_state
        self._mapper = _fit_entry_display_mapper(
            picture,
            figure,
            analysis,
            tolerance_policy=tolerance_policy,
        )
        if geometry_rig_state is None:
            if projection is None:
                raise TikzNativeOpenFaceVisibility3DManimError(
                    "a parallel projection is required without geometry_rig_state"
                )
            self.projection = projection
            self.display_point_provider = display_point_provider
        else:
            project_scene = _geometry_rig_project_scene(geometry_rig_state)
            self.projection = ParallelProjection(
                lambda _scene: _geometry_rig_projection_matrix(
                    project_scene,
                    self._mapper,
                    self.tolerance_policy,
                )
            )
            self.display_point_provider = project_scene
        self.sources = _managed_sources(figure, analysis)
        self.sources.update(
            _geometry_rig_relation_sources(picture, figure, geometry_rig_state)
        )
        _validate_runtime_sources(
            picture,
            figure,
            analysis,
            self._mapper,
            tolerance_policy=tolerance_policy,
        )
        width_scale = _stroke_width_per_pt(picture, figure, analysis)
        binding_map = {item.source_edge_id: item for item in analysis.stroke_bindings}
        self.styles = {
            stroke.source_edge_id: _edge_occlusion_style(
                binding_map[stroke.source_edge_id],
                capacity_style=capacity_style,
                mapper=self._mapper,
            )
            for stroke in self.model.strokes
        }
        self.capacities = {
            stroke.source_edge_id: _capacity_for(
                stroke, self.model, self.styles[stroke.source_edge_id]
            )
            for stroke in self.model.strokes
        }
        _guard_realtime_scale(self.model, self.capacities)

        # The scale gate above must run before the first proxy, slot, or VGroup
        # allocation.  Large authoring payloads therefore cannot cause a
        # partially allocated realtime binding.
        self.proxies = {
            edge_id: _complete_proxy_line(
                binding_map[edge_id], stroke_width_per_pt=width_scale
            )
            for edge_id in sorted(binding_map)
        }
        self.resolved_styles = {
            edge_id: self.styles[edge_id].resolve_for(self.proxies[edge_id])
            for edge_id in sorted(self.proxies)
        }
        self._slots = {
            edge_id: _StrokeSlots(self.capacities[edge_id])
            for edge_id in sorted(self.capacities)
        }
        self.overlay_root = VGroup(
            *(self._slots[edge_id].root for edge_id in sorted(self._slots))
        )

        def update_overlay(mobject: Mobject, dt: float) -> None:
            del mobject
            if self._attached:
                self.update(dt)

        self.overlay_root.add_updater(update_overlay)
        self._attached = False
        self._owner_claimed = False
        self._snapshots: dict[str, tuple[object, ...]] = {}
        self._fixed_frame_camera: ThreeDCamera | None = None
        self.last_frame: OpenFaceVisibilityFrame | None = None

    @property
    def attached(self) -> bool:
        return self._attached

    def _current_inputs(
        self,
    ) -> tuple[dict[str, np.ndarray], tuple[tuple[float, float, float], ...]]:
        raw = self.position_provider()
        positions = {
            key: _point3(value, f"vertex {key}") for key, value in raw.items()
        }
        if set(positions) != set(self.model.vertex_map):
            missing = sorted(set(self.model.vertex_map) - set(positions))
            extra = sorted(set(positions) - set(self.model.vertex_map))
            raise TikzNativeOpenFaceVisibility3DManimError(
                "vertex position identity mismatch"
                + (f"; missing={missing}" if missing else "")
                + (f"; extra={extra}" if extra else "")
            )
        try:
            projection = self.projection.current_matrix(self.scene)
        except ValueError as exc:
            raise TikzNativeOpenFaceVisibility3DManimError(str(exc)) from exc
        return positions, projection

    def _display_point(
        self,
        point: Sequence[float],
        projection: Sequence[Sequence[float]],
    ) -> np.ndarray:
        if self.display_point_provider is not None:
            return _point3(self.display_point_provider(point), "display point")
        return self._mapper.map_point(point, projection)

    def _validate_live_sources(
        self,
        endpoints: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        bindings = {
            item.source_edge_id: item for item in self.analysis.stroke_bindings
        }
        specs = {item.id: item for item in self.picture.objects}
        proofs_by_edge: dict[str, list[LegacyRelationProof3D]] = {}
        for proof in self.analysis.relation_proofs:
            proofs_by_edge.setdefault(proof.source_edge_id, []).append(proof)

        for stroke in self.model.strokes:
            edge_id = stroke.source_edge_id
            full_start, full_end = endpoints[edge_id]
            proofs = proofs_by_edge.get(edge_id, [])
            if not proofs:
                binding = bindings[edge_id]
                if len(binding.object_ids) != 1:
                    raise TikzNativeOpenFaceVisibility3DManimError(
                        f"plain stroke {edge_id} lost its unique source"
                    )
                object_id = binding.object_ids[0]
                source = self.figure.objects.get(object_id)
                if source is None:
                    raise TikzNativeOpenFaceVisibility3DManimError(
                        f"plain stroke {edge_id} source is missing"
                    )
                _validate_live_line_family(
                    source,
                    full_start,
                    full_end,
                    label=f"plain stroke {edge_id}",
                    tolerance_policy=self.tolerance_policy,
                    mapper_tolerance=self._mapper.residual_tolerance,
                    require_complete_line=True,
                    active_only=False,
                    allow_empty=False,
                )
                continue

            for proof in proofs:
                if self._geometry_rig_state is None:
                    full_length = float(np.linalg.norm(full_end - full_start))
                    resolved = self.tolerance_policy.resolve(
                        (full_start, full_end), edge_length=full_length
                    )
                    empty_threshold = _NATIVE_DASH_EMPTY_THRESHOLD + max(
                        self._mapper.residual_tolerance,
                        resolved.boundary,
                    )
                    sources_with_empty_policy: tuple[tuple[Mobject | None, bool], ...] = tuple(
                        (
                            self.figure.objects.get(fragment.object_id),
                            bool(specs[fragment.object_id].style.dash_pattern_pt)
                            and (
                                fragment.end_parameter - fragment.start_parameter
                            )
                            * full_length
                            <= empty_threshold,
                        )
                        for fragment in proof.fragments
                    )
                    if any(source is None for source, _allow in sources_with_empty_policy):
                        raise TikzNativeOpenFaceVisibility3DManimError(
                            f"relation {proof.relation_id} lost a proven source fragment"
                        )
                else:
                    sources_with_empty_policy = (
                        (
                            self.sources.get(
                                f"geometry-rig-relation:{proof.relation_id}"
                            ),
                            False,
                        ),
                    )
                    if sources_with_empty_policy[0][0] is None:
                        raise TikzNativeOpenFaceVisibility3DManimError(
                            f"relation {proof.relation_id} lost its Geometry Rig group"
                        )
                for source, allow_empty in sources_with_empty_policy:
                    assert isinstance(source, Mobject)
                    _validate_live_line_family(
                        source,
                        full_start,
                        full_end,
                        label=f"relation {proof.relation_id}",
                        tolerance_policy=self.tolerance_policy,
                        mapper_tolerance=self._mapper.residual_tolerance,
                        require_complete_line=False,
                        active_only=self._geometry_rig_state is not None,
                        allow_empty=allow_empty,
                    )

    def _prepare_frame(
        self,
    ) -> tuple[
        OpenFaceVisibilityFrame,
        dict[str, OverlayPlan],
        dict[str, tuple[np.ndarray, np.ndarray]],
    ]:
        positions, projection = self._current_inputs()
        endpoints: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for stroke in self.model.strokes:
            start = self._display_point(positions[stroke.vertex_ids[0]], projection)
            end = self._display_point(positions[stroke.vertex_ids[1]], projection)
            endpoints[stroke.source_edge_id] = (start, end)
        self._validate_live_sources(endpoints)
        frame = compute_open_face_visibility(
            self.model,
            projection_matrix=projection,
            vertex_positions=positions,
            tolerance_policy=self.tolerance_policy,
        )
        plans: dict[str, OverlayPlan] = {}
        for stroke in self.model.strokes:
            start, end = endpoints[stroke.source_edge_id]
            plans[stroke.source_edge_id] = build_overlay_plan(
                frame.edge_map[stroke.source_edge_id],
                display_start=start,
                display_end=end,
                capacity=self.capacities[stroke.source_edge_id],
                style=self.styles[stroke.source_edge_id],
            )
        if set(frame.advisory_face_draw_order) != set(self.model.face_map):
            raise TikzNativeOpenFaceVisibility3DManimError(
                "open-face trace draw order does not cover every managed face"
            )
        return frame, plans, endpoints

    def _apply_frame(
        self,
        frame: OpenFaceVisibilityFrame,
        plans: Mapping[str, OverlayPlan],
        endpoints: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        for edge_id in sorted(self._slots):
            start, end = endpoints[edge_id]
            # ``Mobject.put_start_and_end_on`` cannot expand a Line whose
            # previous frame was exactly end-on and therefore zero-length.
            # Rebuild the off-scene proxy directly, and prime only degenerate
            # stable slot Lines before the frozen slot writer updates them.
            self.proxies[edge_id].set_points_by_ends(start, end, buff=0)
            slots = self._slots[edge_id]
            for line in (
                *slots.visible,
                *(item for group in slots.hidden for item in group),
            ):
                if float(np.linalg.norm(line.get_end() - line.get_start())) <= (
                    self.tolerance_policy.absolute_floor
                ):
                    line.set_points_by_ends((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), buff=0)
            slots.apply(plans[edge_id], self.resolved_styles[edge_id])
        self.last_frame = frame

    def _remove_overlay_identity(self) -> None:
        family_ids = {id(item) for item in self.overlay_root.get_family()}
        for container in _scene_containers(self.scene):
            container[:] = [item for item in container if id(item) not in family_ids]

    def _register_fixed_frame(self) -> None:
        camera = getattr(self.scene, "camera", None)
        if isinstance(camera, ThreeDCamera):
            self._fixed_frame_camera = camera
            camera.add_fixed_in_frame_mobjects(self.overlay_root)

    def _remove_fixed_frame(self) -> None:
        if self._fixed_frame_camera is not None:
            self._fixed_frame_camera.remove_fixed_in_frame_mobjects(self.overlay_root)
            self._fixed_frame_camera = None

    def _invalidate_static_image(self) -> None:
        renderer = getattr(self.scene, "renderer", None)
        if renderer is not None and hasattr(renderer, "static_image"):
            renderer.static_image = None

    def attach(self) -> "OpenFaceManimBinding3D":
        if self._attached:
            return self
        if not _using_cairo_renderer():
            raise TikzNativeOpenFaceVisibility3DManimError(
                "open-face automatic visibility v1 supports the Cairo renderer only"
            )
        if any(
            any(item is self.overlay_root for item in container)
            for container in _scene_containers(self.scene)
        ):
            raise TikzNativeOpenFaceVisibility3DManimError(
                "overlay root is already owned by the Scene"
            )
        _claim_figure_owner(self)
        snapshots: dict[str, tuple[object, ...]] = {}
        try:
            frame, plans, endpoints = self._prepare_frame()
            z_indices = _surrogate_z_indices(
                self.scene, self.picture, self.figure, self.analysis, self.sources
            )
            snapshots = {
                object_id: _capture_family_style(source)
                for object_id, source in self.sources.items()
            }
            for edge_id in sorted(self._slots):
                self._slots[edge_id].root.set_z_index(z_indices[edge_id], family=True)
                self._slots[edge_id].apply_static_style(self.resolved_styles[edge_id])
            self._apply_frame(frame, plans, endpoints)
            for object_id in sorted(snapshots):
                _hide_snapshots(snapshots[object_id])
            self._snapshots = snapshots  # type: ignore[assignment]
            self._attached = True
            self.scene.mobjects.append(self.overlay_root)
            self._register_fixed_frame()
            self._invalidate_static_image()
        except Exception:
            try:
                self._attached = False
                for values in snapshots.values():
                    _restore_snapshots(values)
                self._remove_fixed_frame()
                self._remove_overlay_identity()
                self._snapshots = {}
                self._invalidate_static_image()
            finally:
                _release_figure_owner(self)
            raise
        return self

    def update(self, dt: float = 0.0) -> "OpenFaceManimBinding3D":
        del dt
        if not self._attached:
            raise TikzNativeOpenFaceVisibility3DManimError(
                "open-face visibility binding is not attached"
            )
        try:
            frame, plans, endpoints = self._prepare_frame()
            self._apply_frame(frame, plans, endpoints)
        finally:
            # Geometry Rig updaters may rebuild a fragment's stroke RGBA while
            # moving its points.  Suppression is a lifecycle invariant even if
            # this frame's solver/capacity preparation fails and the overlay
            # intentionally remains at its last-good frame.
            for object_id in sorted(self._snapshots):
                _hide_snapshots(self._snapshots[object_id])
        return self

    def restore(self) -> "OpenFaceManimBinding3D":
        self._attached = False
        try:
            self._remove_fixed_frame()
            self._remove_overlay_identity()
            # If the enclosing Geometry Rig has already restored the original
            # ShapeState, its entry snapshots are authoritative.  Reapplying
            # styles captured while the rig was active would re-hide those
            # restored compiler fragments.  In the recommended nesting order
            # (visibility first, rig second) the rig is still active here, so
            # the binding restores only its own temporary suppression.
            if not _geometry_rig_is_restored(self._geometry_rig_state):
                for object_id in sorted(self._snapshots):
                    _restore_snapshots(self._snapshots[object_id])
            self._snapshots = {}
            self._invalidate_static_image()
        finally:
            if self._owner_claimed:
                _release_figure_owner(self)
        return self

    @contextmanager
    def session(self) -> Iterator["OpenFaceManimBinding3D"]:
        self.attach()
        try:
            yield self
        finally:
            self.restore()

    def slot_identities(self) -> tuple[int, ...]:
        return tuple(
            identity
            for edge_id in sorted(self._slots)
            for identity in self._slots[edge_id].identities()
        )

    def slot_snapshot(self) -> tuple[object, ...]:
        values: list[object] = []
        for edge_id in sorted(self._slots):
            for member in self._slots[edge_id].root.get_family():
                points = np.asarray(member.get_all_points(), dtype=float)
                values.append(tuple(np.round(points.reshape(-1), 12)))
                rgbas = getattr(member, "stroke_rgbas", np.empty((0, 4)))
                values.append(tuple(np.round(np.asarray(rgbas).reshape(-1), 12)))
        return tuple(values)


@dataclass(frozen=True)
class TikzNativeOpenFaceAutoOcclusion3D:
    analysis: TikzNativeOpenFaceVisibility3DAdapterResult
    controller: OpenFaceManimBinding3D

    @property
    def last_frame(self) -> OpenFaceVisibilityFrame | None:
        return self.controller.last_frame

    def attach(self) -> "TikzNativeOpenFaceAutoOcclusion3D":
        self.controller.attach()
        return self

    def update(self, dt: float = 0.0) -> "TikzNativeOpenFaceAutoOcclusion3D":
        self.controller.update(dt)
        return self

    def restore(self) -> "TikzNativeOpenFaceAutoOcclusion3D":
        self.controller.restore()
        return self

    def session(self) -> ContextManager[OpenFaceManimBinding3D]:
        return self.controller.session()


def bind_picture_open_face_visibility_3d(
    scene: object,
    picture: PictureSpec,
    figure: NativeFigure,
    *,
    style: OcclusionStyle,
    coordinate_provider: CoordinateProvider | None = None,
    projection: ParallelProjection | None = None,
    display_point_provider: DisplayPointProvider | None = None,
    geometry_rig_state: Mapping[str, object] | None = None,
    default_hidden_style: Mapping[str, object] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> TikzNativeOpenFaceAutoOcclusion3D:
    """Prove and bind TikZ open faces under the adapter's fixed v1 tolerance."""

    policy = TolerancePolicy()
    if geometry_rig_state is not None:
        conflicts = tuple(
            label
            for label, value in (
                ("coordinate_provider", coordinate_provider),
                ("projection", projection),
                ("display_point_provider", display_point_provider),
            )
            if value is not None
        )
        if conflicts:
            raise TikzNativeOpenFaceVisibility3DManimError(
                "geometry_rig_state is mutually exclusive with " + ", ".join(conflicts)
            )
        state_coordinates = geometry_rig_state.get("coordinates")
        if not callable(state_coordinates):
            raise TikzNativeOpenFaceVisibility3DManimError(
                "geometry_rig_state has no live coordinates provider"
            )
        coordinate_provider = state_coordinates
    analysis = adapt_picture_open_face_visibility_3d(
        picture,
        default_hidden_style=default_hidden_style,
        overrides=overrides,
    )
    positions = _canonical_position_provider(
        analysis, coordinate_provider, tolerance_policy=policy
    )
    current_projection = projection or ParallelProjection(analysis.entry_projection)
    controller = OpenFaceManimBinding3D(
        scene,
        picture,
        figure,
        analysis,
        position_provider=positions,
        projection=current_projection,
        display_point_provider=display_point_provider,
        capacity_style=style,
        tolerance_policy=policy,
        geometry_rig_state=geometry_rig_state,
    )
    return TikzNativeOpenFaceAutoOcclusion3D(analysis, controller)


__all__ = [
    "CoordinateProvider",
    "OpenFaceManimBinding3D",
    "TikzNativeOpenFaceAutoOcclusion3D",
    "TikzNativeOpenFaceVisibility3DManimError",
    "bind_picture_open_face_visibility_3d",
]
