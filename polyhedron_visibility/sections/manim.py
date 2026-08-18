from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Callable, Mapping, Sequence

import numpy as np
from manim import Dot, Line, Mobject, Polygon, VGroup

from ..api import ParallelProjection
from ..binding import (
    DisplayPointProvider,
    ManimOcclusionBinding,
    OcclusionBindingError,
    OverlayCapacity,
    OverlayPlan,
    PositionProvider,
    _StrokeSlots,
    build_overlay_plan,
)
from ..contract import StrokeSpec, TolerancePolicy, VertexSpec, VisibilityModel
from ..depth_cue import (
    FaceDepthCueFrame,
    FaceDepthCueLayer,
    FaceDepthCueStyle,
    compute_face_depth_cue,
)
from ..depth_cue.manim import depth_cued_stroke_style
from ..parallel_solver import compute_frame_visibility
from ..style import OcclusionStyle, ResolvedOcclusionStyle
from ..trace import VisibilityFrame
from .contract import SectionPlane3D
from .compositing import (
    TransparentSectionCompositingFrame,
    compute_transparent_section_compositing,
)
from .compositing_manim import (
    PreparedTransparentSectionFrame,
    TransparentSectionLayer,
)
from .solver import (
    ConvexSectionSolverError,
    compute_sectioned_visibility,
    fit_plane_patch_to_convex_polyhedron,
    intersect_plane_with_convex_polyhedron,
    intersect_segment_with_convex_polyhedron,
)
from .trace import (
    ConvexSectionFrame,
    NamedStrokeSolidIntersection,
    SectionedVisibilityFrame,
)


class ConvexSectionManimError(OcclusionBindingError):
    """Raised before a section binding mutates source or Scene state."""


class ConvexSectionBindingScaleError(ConvexSectionManimError):
    """Raised before allocation when a realtime section model is too large."""


PlaneProvider = Callable[[], SectionPlane3D]
PLANE_PATCH_MODES = frozenset({"auto", "strict"})


@dataclass(frozen=True)
class ConvexSectionBindingScaleLimits:
    max_faces: int = 64
    max_strokes: int = 128
    max_surface_edges: int = 192
    max_candidate_pairs: int = 8192
    max_overlay_line_slots: int = 65536


CONVEX_SECTION_BINDING_SCALE_LIMITS = ConvexSectionBindingScaleLimits()


def _finite_non_negative(value: float, label: str) -> float:
    result = float(value)
    if not isfinite(result) or result < 0:
        raise ConvexSectionManimError(f"{label} must be finite and non-negative")
    return result


def _finite_positive(value: float, label: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise ConvexSectionManimError(f"{label} must be finite and positive")
    return result


@dataclass(frozen=True)
class ConvexSectionStyle:
    plane_fill_color: object = "#8CC8C0"
    plane_fill_opacity: float = 0.12
    plane_stroke_color: object = "#2A9D8F"
    plane_stroke_width: float = 1.5
    section_fill_color: object = "#F4C95D"
    section_fill_opacity: float = 0.38
    boundary_color: object = "#D97706"
    boundary_hidden_color: object = "#B45309"
    boundary_width: float = 4.0
    boundary_hidden_width: float = 3.0
    boundary_hidden_opacity: float = 0.78
    point_color: object = "#B91C1C"
    point_radius: float = 0.045
    intersection_point_color: object = "#7C3AED"
    intersection_point_radius: float = 0.055
    max_boundary_projected_length: float = 12.0
    dash_length: float = 0.10
    dash_gap: float = 0.07
    show_plane: bool = True
    show_points: bool = True
    show_intersection_points: bool = True

    def __post_init__(self) -> None:
        for name in (
            "plane_fill_opacity",
            "section_fill_opacity",
            "boundary_hidden_opacity",
        ):
            value = _finite_non_negative(getattr(self, name), name)
            if value > 1:
                raise ConvexSectionManimError(f"{name} must not exceed 1")
            object.__setattr__(self, name, value)
        for name in (
            "plane_stroke_width",
            "boundary_width",
            "boundary_hidden_width",
            "point_radius",
            "intersection_point_radius",
            "max_boundary_projected_length",
            "dash_length",
        ):
            object.__setattr__(
                self, name, _finite_positive(getattr(self, name), name)
            )
        object.__setattr__(
            self, "dash_gap", _finite_non_negative(self.dash_gap, "dash_gap")
        )
        if (
            not isinstance(self.show_plane, bool)
            or not isinstance(self.show_points, bool)
            or not isinstance(self.show_intersection_points, bool)
        ):
            raise ConvexSectionManimError(
                "show_plane, show_points, and show_intersection_points must be boolean"
            )


@dataclass(frozen=True)
class _PreparedSection:
    trace: SectionedVisibilityFrame
    transparent_compositing: TransparentSectionCompositingFrame | None
    prepared_transparent: PreparedTransparentSectionFrame | None
    face_depth_cue: FaceDepthCueFrame | None
    face_display_points: Mapping[str, np.ndarray]
    boundary_plans: tuple[OverlayPlan, ...]
    plane_display_points: tuple[tuple[float, float, float], ...]
    section_display_points: tuple[tuple[float, float, float], ...]
    stroke_intersection_display_points: tuple[
        tuple[str, tuple[tuple[float, float, float], ...]], ...
    ]
    display_plane: SectionPlane3D


def _surface_edges(model: VisibilityModel) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                tuple(
                    sorted(
                        (
                            start,
                            face.vertex_ids[(index + 1) % len(face.vertex_ids)],
                        )
                    )
                )
                for face in model.faces
                for index, start in enumerate(face.vertex_ids)
            }
        )
    )


def _capacity(
    candidate_count: int,
    style: OcclusionStyle,
    *,
    always_hidden: bool = False,
) -> OverlayCapacity:
    hidden = max(1, candidate_count) if always_hidden else candidate_count
    return OverlayCapacity(
        visible_slots=candidate_count + 1,
        hidden_slots=hidden,
        dash_slots_per_hidden=(
            int(ceil(style.max_projected_length / style.dash_period)) + hidden + 1
        ),
        max_projected_length=style.max_projected_length,
    )


def _empty_plan() -> OverlayPlan:
    return OverlayPlan((), ())


def _guard_realtime_scale(
    model: VisibilityModel,
    source_style: OcclusionStyle,
    section_style: ConvexSectionStyle,
) -> None:
    limits = CONVEX_SECTION_BINDING_SCALE_LIMITS
    surface_edge_count = len(_surface_edges(model))
    fixed = (
        ("faces", len(model.faces), limits.max_faces),
        ("strokes", len(model.strokes), limits.max_strokes),
        ("surface_edges", surface_edge_count, limits.max_surface_edges),
    )
    for label, count, maximum in fixed:
        if count > maximum:
            raise ConvexSectionBindingScaleError(
                f"convex-section realtime binding {label}={count} "
                f"exceeds fixed v1 limit {maximum}"
            )

    source_dash_base = int(
        ceil(source_style.max_projected_length / source_style.dash_period)
    )
    boundary_dash_base = int(
        ceil(
            section_style.max_boundary_projected_length
            / (section_style.dash_length + section_style.dash_gap)
        )
    )
    candidate_pairs = 0
    overlay_line_slots = 0
    for stroke in model.strokes:
        candidates = sum(
            1
            for face in model.faces
            if face.occludes_strokes
            and face.face_id not in stroke.incident_face_ids
        )
        if stroke.visibility_mode == "auto":
            candidates += 1
        candidate_pairs += candidates
        hidden = max(1, candidates) if stroke.visibility_mode == "always_hidden" else candidates
        overlay_line_slots += candidates + 1
        overlay_line_slots += hidden * (source_dash_base + hidden + 1)

    boundary_candidates = len(model.faces)
    candidate_pairs += surface_edge_count * boundary_candidates
    boundary_hidden = boundary_candidates
    overlay_line_slots += surface_edge_count * (
        boundary_candidates
        + 1
        + boundary_hidden
        * (boundary_dash_base + boundary_hidden + 1)
    )
    if candidate_pairs > limits.max_candidate_pairs:
        raise ConvexSectionBindingScaleError(
            "convex-section realtime binding candidate_pairs="
            f"{candidate_pairs} exceeds fixed v1 limit "
            f"{limits.max_candidate_pairs}"
        )
    if overlay_line_slots > limits.max_overlay_line_slots:
        raise ConvexSectionBindingScaleError(
            "convex-section realtime binding overlay_line_slots="
            f"{overlay_line_slots} exceeds fixed v1 limit "
            f"{limits.max_overlay_line_slots}"
        )


def _point3(value: Sequence[float], label: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ConvexSectionManimError(
            f"{label} must be a finite three-component point"
        )
    return point


def _boundary_model(
    solid: VisibilityModel,
    positions: Mapping[str, np.ndarray],
    section: ConvexSectionFrame,
) -> tuple[VisibilityModel, dict[str, np.ndarray]] | None:
    if not section.boundary_segments:
        return None
    point_vertex_ids: dict[str, str] = {}
    extra_vertices: dict[str, np.ndarray] = {}
    for point in section.points:
        if point.source_vertex_ids:
            vertex_id = point.source_vertex_ids[0]
        else:
            vertex_id = f"__section_point__:{point.point_id}"
            extra_vertices[vertex_id] = np.asarray(point.position, dtype=float)
        point_vertex_ids[point.point_id] = vertex_id
    vertices = [
        VertexSpec(
            vertex.vertex_id,
            tuple(float(item) for item in positions[vertex.vertex_id]),
        )
        for vertex in solid.vertices
    ]
    vertices.extend(
        VertexSpec(vertex_id, tuple(float(item) for item in point))
        for vertex_id, point in sorted(extra_vertices.items())
    )
    strokes = tuple(
        StrokeSpec(
            segment.segment_id,
            (
                point_vertex_ids[segment.start_point_id],
                point_vertex_ids[segment.end_point_id],
            ),
            (),
            "auto",
        )
        for segment in section.boundary_segments
    )
    model = VisibilityModel(
        f"{solid.visibility_group_id}:section-boundary",
        tuple(sorted(vertices, key=lambda item: item.vertex_id)),
        solid.faces,
        strokes,
    )
    current = dict(positions)
    current.update(extra_vertices)
    return model, current


class ConvexSection3D(ManimOcclusionBinding):
    """Stable Cairo binding for one infinite plane and one convex solid.

    Registered solid/free strokes are solved against every solid face and the
    automatically fitted display patch.  The authored patch dimensions are
    minimums by default; use ``plane_patch_mode='strict'`` to retain literal
    finite-patch behavior.  The derived section polygon, boundary slots, and
    point markers are preallocated once, so triangle/quad/hexagon/empty
    transitions never replace Scene objects during ``play``.  With
    ``accurate_transparency=True``, the solid fills and fitted display patch
    are also split into a fixed triangle pool and locally sorted far to near.
    """

    def __init__(
        self,
        scene: object,
        model: VisibilityModel,
        *,
        position_provider: PositionProvider,
        stroke_bindings: Mapping[str, Mobject],
        plane_provider: PlaneProvider,
        projection: ParallelProjection,
        source_style: OcclusionStyle,
        section_style: ConvexSectionStyle | None = None,
        face_fill_bindings: Mapping[str, Mobject] | None = None,
        face_depth_style: FaceDepthCueStyle | None = None,
        accurate_transparency: bool = False,
        transparent_coplanar_policy: str = "section_over_solid",
        plane_patch_mode: str = "auto",
        plane_patch_margin: float = 0.15,
        display_point_provider: DisplayPointProvider | None = None,
        tolerance_policy: TolerancePolicy | None = None,
        source_coordinate_mode: str = "world",
        section_id: str = "section",
    ) -> None:
        if not isinstance(section_id, str) or not section_id.strip():
            raise ConvexSectionManimError(
                "section_id must be a non-empty string"
            )
        self.section_id = section_id
        self.plane_provider = plane_provider
        self.projection = projection
        self.section_style = section_style or ConvexSectionStyle()
        if not isinstance(accurate_transparency, bool):
            raise ConvexSectionManimError(
                "accurate_transparency must be boolean"
            )
        if accurate_transparency and face_fill_bindings is None:
            raise ConvexSectionManimError(
                "accurate_transparency requires face_fill_bindings for every solid face"
            )
        if transparent_coplanar_policy not in {
            "section_over_solid",
            "solid_over_section",
            "fail",
        }:
            raise ConvexSectionManimError(
                "transparent_coplanar_policy must be 'section_over_solid', "
                "'solid_over_section', or 'fail'"
            )
        self.accurate_transparency = accurate_transparency
        self.transparent_coplanar_policy = transparent_coplanar_policy
        if plane_patch_mode not in PLANE_PATCH_MODES:
            raise ConvexSectionManimError(
                "plane_patch_mode must be 'auto' or 'strict'"
            )
        self.plane_patch_mode = plane_patch_mode
        self.plane_patch_margin = _finite_non_negative(
            plane_patch_margin, "plane_patch_margin"
        )
        if face_depth_style is not None and face_fill_bindings is None:
            raise ConvexSectionManimError(
                "face_depth_style requires face_fill_bindings"
            )
        self.face_depth_style = face_depth_style or FaceDepthCueStyle()
        _guard_realtime_scale(model, source_style, self.section_style)
        self._initial_plane = self._current_plane()
        self._plane_contract = (
            self._initial_plane.plane_id,
            self._initial_plane.half_width,
            self._initial_plane.half_height,
            self._initial_plane.occludes_strokes,
        )
        self._auto_patch_half_width = self._initial_plane.half_width
        self._auto_patch_half_height = self._initial_plane.half_height
        self.last_display_plane: SectionPlane3D | None = None
        self.last_sectioned_frame: SectionedVisibilityFrame | None = None
        self.last_face_depth_cue: FaceDepthCueFrame | None = None
        self.last_transparent_compositing: (
            TransparentSectionCompositingFrame | None
        ) = None
        self._prepared_section: _PreparedSection | None = None

        super().__init__(
            scene,
            model,
            position_provider=position_provider,
            stroke_bindings=stroke_bindings,
            projection_provider=projection.current_matrix,
            display_point_provider=display_point_provider,
            style=source_style,
            tolerance_policy=tolerance_policy,
            require_closed_convex_manifold=True,
            source_coordinate_mode=source_coordinate_mode,
        )
        self._transparent_layer = (
            TransparentSectionLayer(
                model,
                face_fill_bindings,
                tolerance_policy=self.tolerance_policy,
                source_coordinate_mode=source_coordinate_mode,  # type: ignore[arg-type]
            )
            if accurate_transparency and face_fill_bindings is not None
            else None
        )
        self._face_depth_layer = (
            FaceDepthCueLayer(
                model,
                face_fill_bindings,
                tolerance_policy=self.tolerance_policy,
                source_coordinate_mode=source_coordinate_mode,  # type: ignore[arg-type]
            )
            if face_fill_bindings is not None and not accurate_transparency
            else None
        )

        extra_face = 1 if self._initial_plane.occludes_strokes else 0
        self.capacities = {}
        for stroke in model.strokes:
            candidates = sum(
                1
                for face in model.faces
                if face.occludes_strokes
                and face.face_id not in stroke.incident_face_ids
            )
            if stroke.visibility_mode == "auto":
                candidates += extra_face
            self.capacities[stroke.source_edge_id] = _capacity(
                candidates,
                source_style,
                always_hidden=(stroke.visibility_mode == "always_hidden"),
            )
        self._slots = {
            stroke.source_edge_id: _StrokeSlots(
                self.capacities[stroke.source_edge_id]
            )
            for stroke in model.strokes
        }

        boundary_style = OcclusionStyle(
            max_projected_length=self.section_style.max_boundary_projected_length,
            dash_length=self.section_style.dash_length,
            dash_gap=self.section_style.dash_gap,
            visible_color=self.section_style.boundary_color,
            hidden_color=self.section_style.boundary_hidden_color,
            visible_width_scale=1.0,
            hidden_width_scale=(
                self.section_style.boundary_hidden_width
                / self.section_style.boundary_width
            ),
            hidden_opacity_scale=self.section_style.boundary_hidden_opacity,
        )
        prototype = Line((0, 0, 0), (1, 0, 0), buff=0)
        prototype.set_stroke(
            color=self.section_style.boundary_color,
            width=self.section_style.boundary_width,
            opacity=1.0,
        )
        self._boundary_style_spec = boundary_style
        self._resolved_boundary_style: ResolvedOcclusionStyle = (
            boundary_style.resolve_for(prototype)
        )
        boundary_capacity = _capacity(len(model.faces), boundary_style)
        self._boundary_capacities = tuple(
            boundary_capacity for _item in _surface_edges(model)
        )
        self._boundary_slots = tuple(
            _StrokeSlots(capacity) for capacity in self._boundary_capacities
        )
        for slot in self._boundary_slots:
            slot.apply_static_style(self._resolved_boundary_style)

        dummy = ((0, 0, 0), (1, 0, 0), (0, 1, 0))
        self.plane_patch = Polygon(*dummy)
        self.section_fill = Polygon(*dummy)
        self._point_slots = tuple(
            Dot(radius=self.section_style.point_radius)
            for _item in _surface_edges(model)
        )
        self._intersection_point_slots = {
            stroke.source_edge_id: (
                Dot(radius=self.section_style.intersection_point_radius),
                Dot(radius=self.section_style.intersection_point_radius),
            )
            for stroke in model.strokes
            if not stroke.incident_face_ids
        }
        self.boundary_root = VGroup(
            *(item.root for item in self._boundary_slots)
        )
        self.point_root = VGroup(*self._point_slots)
        self.intersection_point_root = VGroup(
            *(
                dot
                for edge_id in sorted(self._intersection_point_slots)
                for dot in self._intersection_point_slots[edge_id]
            )
        )
        overlay_members: list[Mobject] = []
        if self._transparent_layer is not None:
            overlay_members.append(self._transparent_layer.root)
        elif self._face_depth_layer is not None:
            overlay_members.append(self._face_depth_layer.root)
        overlay_members.extend((
            self.plane_patch,
            self.section_fill,
            *(self._slots[key].root for key in sorted(self._slots)),
            self.boundary_root,
            self.point_root,
            self.intersection_point_root,
        ))
        self.overlay_root = VGroup(*overlay_members)

        def update_overlay(mobject: object, dt: float) -> None:
            del mobject
            if self._attached:
                self.update(dt)

        self.overlay_root.add_updater(update_overlay)

    def _current_plane(self) -> SectionPlane3D:
        value = self.plane_provider()
        if not isinstance(value, SectionPlane3D):
            raise ConvexSectionManimError(
                "plane_provider must return SectionPlane3D"
            )
        return value

    def _validated_plane(self) -> SectionPlane3D:
        plane = self._current_plane()
        contract = (
            plane.plane_id,
            plane.half_width,
            plane.half_height,
            plane.occludes_strokes,
        )
        if contract != self._plane_contract:
            raise ConvexSectionManimError(
                "plane identity, patch size, and occlusion policy must stay fixed"
            )
        return plane

    def _display_plane(
        self,
        plane: SectionPlane3D,
        positions: Mapping[str, Sequence[float]],
    ) -> SectionPlane3D:
        if self.plane_patch_mode == "strict":
            return plane
        try:
            fitted = fit_plane_patch_to_convex_polyhedron(
                self.model,
                plane,
                vertex_positions=positions,
                margin_ratio=self.plane_patch_margin,
                tolerance_policy=self.tolerance_policy,
            )
        except ConvexSectionSolverError as exc:
            raise ConvexSectionManimError(str(exc)) from exc
        return SectionPlane3D(
            fitted.plane_id,
            fitted.point,
            fitted.normal,
            max(fitted.half_width, self._auto_patch_half_width),
            max(fitted.half_height, self._auto_patch_half_height),
            u_axis=fitted.u_axis,
            occludes_strokes=fitted.occludes_strokes,
        )

    def _display(self, point: Sequence[float]) -> tuple[float, float, float]:
        value = (
            point
            if self.display_point_provider is None
            else self.display_point_provider(point)
        )
        result = _point3(value, "section display point")
        return tuple(float(item) for item in result)

    def _prepare_frame(
        self,
    ) -> tuple[VisibilityFrame, dict[str, OverlayPlan], dict[str, np.ndarray]]:
        positions, projection = self._current_inputs()
        logical_plane = self._validated_plane()
        plane = self._display_plane(logical_plane, positions)
        section = intersect_plane_with_convex_polyhedron(
            self.section_id,
            self.model,
            logical_plane,
            vertex_positions=positions,
            tolerance_policy=self.tolerance_policy,
        )
        for point in section.points:
            u_value, v_value = plane.coordinates_in_plane(point.position)
            tolerance = self.tolerance_policy.resolve(
                (point.position, plane.point)
            ).boundary
            if (
                abs(u_value) > plane.half_width + tolerance
                or abs(v_value) > plane.half_height + tolerance
            ):
                raise ConvexSectionManimError(
                    "cutting-plane patch does not cover the derived section"
                )
        source_frame = compute_sectioned_visibility(
            self.model,
            plane,
            projection_matrix=projection,
            vertex_positions=positions,
            tolerance_policy=self.tolerance_policy,
        )
        source_plans: dict[str, OverlayPlan] = {}
        for stroke in self.model.strokes:
            start = positions[stroke.vertex_ids[0]]
            end = positions[stroke.vertex_ids[1]]
            source_plans[stroke.source_edge_id] = build_overlay_plan(
                source_frame.edge_map[stroke.source_edge_id],
                display_start=self._display(start),
                display_end=self._display(end),
                capacity=self.capacities[stroke.source_edge_id],
                style=self.style,
            )

        boundary_frame: VisibilityFrame | None = None
        boundary_plans: list[OverlayPlan] = []
        boundary = _boundary_model(self.model, positions, section)
        if boundary is not None:
            boundary_model, boundary_positions = boundary
            boundary_frame = compute_frame_visibility(
                boundary_model,
                projection_matrix=projection,
                vertex_positions=boundary_positions,
                tolerance_policy=self.tolerance_policy,
                require_closed_convex_manifold=True,
            )
            point_map = section.point_map
            for index, segment in enumerate(section.boundary_segments):
                start = point_map[segment.start_point_id].position
                end = point_map[segment.end_point_id].position
                boundary_plans.append(
                    build_overlay_plan(
                        boundary_frame.edge_map[segment.segment_id],
                        display_start=self._display(start),
                        display_end=self._display(end),
                        capacity=self._boundary_capacities[index],
                        style=self._boundary_style_spec,
                    )
                )
        if len(boundary_plans) > len(self._boundary_slots):
            raise ConvexSectionManimError(
                "section boundary exceeds the preallocated polyhedron edge capacity"
            )

        stroke_intersections = tuple(
            NamedStrokeSolidIntersection(
                stroke.source_edge_id,
                intersect_segment_with_convex_polyhedron(
                    self.model,
                    positions[stroke.vertex_ids[0]],
                    positions[stroke.vertex_ids[1]],
                    vertex_positions=positions,
                    tolerance_policy=self.tolerance_policy,
                ),
            )
            for stroke in self.model.strokes
            if not stroke.incident_face_ids
        )
        trace = SectionedVisibilityFrame(
            section,
            source_frame,
            boundary_frame,
            stroke_intersections,
        )
        face_depth_cue: FaceDepthCueFrame | None = None
        face_display_points: Mapping[str, np.ndarray] = {}
        transparent_compositing: TransparentSectionCompositingFrame | None = None
        prepared_transparent: PreparedTransparentSectionFrame | None = None
        if self._face_depth_layer is not None or self._transparent_layer is not None:
            display_positions = {
                vertex_id: np.asarray(self._display(point), dtype=float)
                for vertex_id, point in positions.items()
            }
            face_depth_cue = compute_face_depth_cue(
                self.model,
                projection_matrix=source_frame.projection_matrix,
                vertex_positions=positions,
                style=self.face_depth_style,
                tolerance_policy=self.tolerance_policy,
                face_draw_order=tuple(
                    face_id
                    for face_id in source_frame.face_draw_order
                    if face_id in self.model.face_map
                ),
                require_closed_convex_manifold=True,
            )
            if self._face_depth_layer is not None:
                face_display_points = self._face_depth_layer.prepare(
                    face_depth_cue,
                    world_points=positions,
                    display_points=display_positions,
                    containers=self._scene_containers(),
                )
        if self._transparent_layer is not None:
            transparent_compositing = compute_transparent_section_compositing(
                self.section_id,
                self.model,
                plane,
                projection_matrix=projection,
                vertex_positions=positions,
                tolerance_policy=self.tolerance_policy,
                coplanar_policy=self.transparent_coplanar_policy,
            )
            prepared_transparent = self._transparent_layer.prepare(
                transparent_compositing,
                world_points=positions,
                display_point_provider=self._display,
                plane_fill_color=self.section_style.plane_fill_color,
                plane_fill_opacity=(
                    self.section_style.plane_fill_opacity
                    if self.section_style.show_plane
                    else 0.0
                ),
                section_fill_color=self.section_style.section_fill_color,
                section_fill_opacity=self.section_style.section_fill_opacity,
                face_depth_cue=face_depth_cue,
                containers=self._scene_containers(),
            )
        self._prepared_section = _PreparedSection(
            trace,
            transparent_compositing,
            prepared_transparent,
            face_depth_cue,
            face_display_points,
            tuple(boundary_plans),
            tuple(self._display(point) for point in plane.patch_corners()),
            tuple(self._display(point.position) for point in section.points),
            tuple(
                (
                    item.source_edge_id,
                    tuple(self._display(hit.position) for hit in item.intersection.hits),
                )
                for item in stroke_intersections
            ),
            plane,
        )
        return source_frame, source_plans, positions

    @staticmethod
    def _set_polygon(
        polygon: Polygon,
        points: Sequence[Sequence[float]],
        *,
        fill_color: object,
        fill_opacity: float,
        stroke_color: object,
        stroke_width: float,
        visible: bool,
    ) -> None:
        if len(points) >= 3:
            values = [np.asarray(item, dtype=float) for item in points]
            polygon.set_points_as_corners([*values, values[0]])
        polygon.set_fill(
            fill_color, opacity=fill_opacity if visible else 0.0
        )
        polygon.set_stroke(
            stroke_color,
            width=stroke_width,
            opacity=1.0 if visible and stroke_width > 0 else 0.0,
        )

    def _apply_frame(
        self,
        frame: VisibilityFrame,
        plans: Mapping[str, OverlayPlan],
    ) -> None:
        prepared = self._prepared_section
        if prepared is None:
            raise ConvexSectionManimError("section frame was not prepared")
        cue_map = (
            None
            if prepared.face_depth_cue is None
            else prepared.face_depth_cue.edge_map
        )
        if self._face_depth_layer is not None:
            if prepared.face_depth_cue is None:
                raise ConvexSectionManimError(
                    "face depth-cue frame was not prepared"
                )
            self._face_depth_layer.apply(
                prepared.face_depth_cue,
                prepared.face_display_points,
            )
        if self._transparent_layer is not None:
            if prepared.prepared_transparent is None:
                raise ConvexSectionManimError(
                    "transparent compositing frame was not prepared"
                )
            self._transparent_layer.apply(prepared.prepared_transparent)
        for edge_id in sorted(self._slots):
            resolved = self._resolved_styles[edge_id]
            if cue_map is not None:
                resolved = depth_cued_stroke_style(
                    resolved, cue_map[edge_id]
                )
            self._slots[edge_id].apply(plans[edge_id], resolved)
        self.last_frame = frame
        for index, slot in enumerate(self._boundary_slots):
            plan = (
                prepared.boundary_plans[index]
                if index < len(prepared.boundary_plans)
                else _empty_plan()
            )
            slot.apply(plan, self._resolved_boundary_style)

        self._set_polygon(
            self.plane_patch,
            prepared.plane_display_points,
            fill_color=self.section_style.plane_fill_color,
            fill_opacity=(
                0.0
                if self._transparent_layer is not None
                else self.section_style.plane_fill_opacity
            ),
            stroke_color=self.section_style.plane_stroke_color,
            stroke_width=self.section_style.plane_stroke_width,
            visible=self.section_style.show_plane,
        )
        section_visible = prepared.trace.section.kind == "polygon"
        self._set_polygon(
            self.section_fill,
            prepared.section_display_points,
            fill_color=self.section_style.section_fill_color,
            fill_opacity=self.section_style.section_fill_opacity,
            stroke_color=self.section_style.section_fill_color,
            stroke_width=0.0,
            visible=(section_visible and self._transparent_layer is None),
        )
        for index, dot in enumerate(self._point_slots):
            if (
                index < len(prepared.section_display_points)
                and self.section_style.show_points
            ):
                dot.move_to(prepared.section_display_points[index])
                dot.set_color(self.section_style.point_color)
                dot.set_opacity(1.0)
            else:
                dot.set_opacity(0.0)
        intersection_points = dict(
            prepared.stroke_intersection_display_points
        )
        for edge_id, slots in self._intersection_point_slots.items():
            points = intersection_points.get(edge_id, ())
            for index, dot in enumerate(slots):
                if (
                    index < len(points)
                    and self.section_style.show_intersection_points
                ):
                    dot.move_to(points[index])
                    dot.set_color(self.section_style.intersection_point_color)
                    dot.set_opacity(1.0)
                else:
                    dot.set_opacity(0.0)
        self.last_sectioned_frame = prepared.trace
        self.last_face_depth_cue = prepared.face_depth_cue
        self.last_transparent_compositing = prepared.transparent_compositing
        self.last_display_plane = prepared.display_plane
        if self.plane_patch_mode == "auto":
            self._auto_patch_half_width = max(
                self._auto_patch_half_width,
                prepared.display_plane.half_width,
            )
            self._auto_patch_half_height = max(
                self._auto_patch_half_height,
                prepared.display_plane.half_height,
            )

    def attach(self) -> "ConvexSection3D":
        if self.attached:
            return self
        super().attach()
        try:
            if self._face_depth_layer is not None:
                self._face_depth_layer.capture_and_hide()
            if self._transparent_layer is not None:
                self._transparent_layer.capture_and_hide()
            source_z = [
                float(item.z_index) for item in self.stroke_bindings.values()
            ]
            minimum = min(source_z, default=10.0)
            maximum = max(source_z, default=10.0)
            self.plane_patch.set_z_index(minimum - 2.0, family=True)
            self.section_fill.set_z_index(minimum - 1.0, family=True)
            self.boundary_root.set_z_index(maximum + 1.0, family=True)
            self.point_root.set_z_index(maximum + 2.0, family=True)
            self.intersection_point_root.set_z_index(
                maximum + 3.0, family=True
            )
            self._invalidate_cairo_static_image()
        except Exception:
            if self._face_depth_layer is not None:
                self._face_depth_layer.restore()
            if self._transparent_layer is not None:
                self._transparent_layer.restore()
            super().restore()
            raise
        return self

    def update(self, dt: float = 0.0) -> "ConvexSection3D":
        try:
            super().update(dt)
        finally:
            if self._face_depth_layer is not None:
                self._face_depth_layer.hide()
            if self._transparent_layer is not None:
                self._transparent_layer.hide()
        return self

    def restore(self) -> "ConvexSection3D":
        try:
            super().restore()
        finally:
            if self._face_depth_layer is not None:
                self._face_depth_layer.restore()
            if self._transparent_layer is not None:
                self._transparent_layer.restore()
            self._prepared_section = None
            self.last_face_depth_cue = None
            self.last_transparent_compositing = None
            self.last_display_plane = None
            self._auto_patch_half_width = self._initial_plane.half_width
            self._auto_patch_half_height = self._initial_plane.half_height
        return self

    def face_fill_identities(self) -> tuple[int, ...]:
        if self._transparent_layer is not None:
            return self._transparent_layer.identities()
        if self._face_depth_layer is None:
            return ()
        return self._face_depth_layer.identities()

    def active_transparent_fragment_ids(self) -> tuple[str, ...]:
        if self._transparent_layer is None:
            return ()
        return self._transparent_layer.active_fragment_ids

    def active_transparent_fragment_z_indices(self) -> dict[str, float]:
        if self._transparent_layer is None:
            return {}
        return self._transparent_layer.active_fragment_z_indices()

    def section_slot_identities(self) -> tuple[int, ...]:
        return (
            id(self.plane_patch),
            id(self.section_fill),
            id(self.boundary_root),
            *(
                identity
                for slot in self._boundary_slots
                for identity in slot.identities()
            ),
            id(self.point_root),
            *(id(item) for item in self._point_slots),
            id(self.intersection_point_root),
            *(
                id(item)
                for edge_id in sorted(self._intersection_point_slots)
                for item in self._intersection_point_slots[edge_id]
            ),
        )

    def intersection_point_identities(self) -> tuple[int, ...]:
        return tuple(
            id(item)
            for edge_id in sorted(self._intersection_point_slots)
            for item in self._intersection_point_slots[edge_id]
        )

    def active_intersection_points(
        self, source_edge_id: str
    ) -> tuple[tuple[float, float, float], ...]:
        if source_edge_id not in self._intersection_point_slots:
            raise ConvexSectionManimError(
                f"stroke {source_edge_id!r} has no free-line intersection slots"
            )
        if (
            self.last_sectioned_frame is None
            or not self.section_style.show_intersection_points
        ):
            return ()
        intersections = {
            item.source_edge_id: item.intersection
            for item in self.last_sectioned_frame.stroke_intersections
        }
        count = len(intersections[source_edge_id].hits)
        return tuple(
            tuple(float(value) for value in dot.get_center())
            for dot in self._intersection_point_slots[source_edge_id][:count]
        )


__all__ = [
    "CONVEX_SECTION_BINDING_SCALE_LIMITS",
    "ConvexSection3D",
    "ConvexSectionBindingScaleError",
    "ConvexSectionBindingScaleLimits",
    "ConvexSectionManimError",
    "ConvexSectionStyle",
    "PlaneProvider",
    "PLANE_PATCH_MODES",
]
