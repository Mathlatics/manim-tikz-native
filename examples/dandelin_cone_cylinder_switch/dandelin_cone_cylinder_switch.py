"""Animate an analytic Dandelin two-sphere cone-to-cylinder limit.

The displayed surface family is

``r(z, p) = R + k(p) z`` with ``k(p) = k0 (1 - p)``.

At ``p = 0`` the lower trim circle collapses to the cone apex.  As ``p``
approaches one, the apex recedes to negative infinity and the generators
become parallel.  At ``p = 1`` the family is exactly a cylinder.  Both sphere
centres, radii, surface-contact circles, and plane-contact points are solved
analytically for every frame.

Preview with::

    PYTHONPATH="$PWD" manim --renderer cairo --disable_caching -ql --fps 12 \
      examples/dandelin_cone_cylinder_switch/dandelin_cone_cylinder_switch.py \
      DandelinConeCylinderSwitch
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import acos, atan, atan2, ceil, cos, isfinite, pi, sin, sqrt
from typing import Sequence

import numpy as np
from manim import (
    DOWN,
    UP,
    Circle,
    DashedLine,
    DashedVMobject,
    Dot,
    FadeIn,
    Line,
    Polygon,
    Scene,
    Text,
    VGroup,
    VMobject,
    ValueTracker,
    always_redraw,
    config,
    smooth,
)

from polyhedron_visibility.compositor import (
    PainterConstraint,
    stable_topological_sort,
)
from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    CylinderSpec,
    PlaneDisplayPatchSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.global_occlusion import (
    compute_global_quadric_frame,
)
from polyhedron_visibility.quadrics.section_compositing import (
    PlaneDepthRole,
    compute_quadric_section_compositing,
    merge_quadric_plane_fragment_contours,
)


BACKGROUND_COLOR = "#0B1622"
SURFACE_RADIUS = 1.45
AXIAL_RANGE = (-3.0, 5.8)
CONE_SLOPE = SURFACE_RADIUS / abs(AXIAL_RANGE[0])
PLANE_X_COMPONENT = 0.25
PLANE_NORMAL = (
    PLANE_X_COMPONENT,
    0.0,
    sqrt(1.0 - PLANE_X_COMPONENT * PLANE_X_COMPONENT),
)
PLANE_OFFSET = 0.20

SURFACE_COLORS = ("#163653", "#214F70", "#2D6B89", "#397F98")
SURFACE_STROKE = "#77E3F2"
PLANE_COLOR = "#31C6AE"
OCCLUDED_PLANE_FILL = "#6B7C93"
OCCLUDED_PLANE_STROKE = "#9FB3C8"
SECTION_COLOR = "#FFD166"
SPHERE_COLOR = "#F59E7A"
SPHERE_STROKE = "#FFD0B8"
FOCUS_COLOR = "#FFF4A3"

PROJECTION_SCALE = 0.65
PROJECTION_OFFSET = np.asarray((0.0, -0.82), dtype=float)
_DEPTH_AXIS = np.asarray((0.75, -1.25, 0.55), dtype=float)
_DEPTH_AXIS /= np.linalg.norm(_DEPTH_AXIS)
_SCREEN_RIGHT = np.asarray((-_DEPTH_AXIS[1], _DEPTH_AXIS[0], 0.0))
_SCREEN_RIGHT /= np.linalg.norm(_SCREEN_RIGHT)
_SCREEN_UP = np.cross(_DEPTH_AXIS, _SCREEN_RIGHT)
_SCREEN_UP /= np.linalg.norm(_SCREEN_UP)
_PROJECTION_MATRIX = np.vstack(
    (
        PROJECTION_SCALE * _SCREEN_RIGHT,
        PROJECTION_SCALE * _SCREEN_UP,
        _DEPTH_AXIS,
    )
)
_PARALLEL_VIEW = ParallelView.from_matrix(_PROJECTION_MATRIX)
_PLANE_POINT = tuple(
    float(PLANE_OFFSET * component) for component in PLANE_NORMAL
)
_SECTION_PLANE = SectionPlane(
    "switch-plane",
    _PLANE_POINT,
    PLANE_NORMAL,
    (0.0, 1.0, 0.0),
)
_PLANE_PATCH = PlaneDisplayPatchSpec(
    "switch-plane-patch",
    _SECTION_PLANE.plane_id,
    3.35,
    3.55,
)
_OCCLUDED_PLANE_ROLES = frozenset(
    {
        PlaneDepthRole.BEHIND_SURFACE,
        PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
    }
)
_SECTION_MAX_SCREEN_ERROR = 0.14
_SECTION_MAX_SEGMENTS = 2048


def _progress(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("switch progress must lie in [0, 1]")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("switch progress must lie in [0, 1]") from exc
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("switch progress must lie in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class DandelinSphereFrame:
    """One sphere and both of its certified tangency records."""

    plane_side: int
    center: tuple[float, float, float]
    radius: float
    surface_contact_radius: float
    surface_contact_z: float
    plane_contact: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class DandelinSwitchFrame:
    """Complete renderer-neutral geometry for one animation progress."""

    progress: float
    slope: float
    apex_z: float | None
    spheres: tuple[DandelinSphereFrame, DandelinSphereFrame]

    @property
    def surface_kind(self) -> str:
        return "cylinder" if self.slope == 0.0 else "cone"

    def radius_at(self, z: float) -> float:
        return SURFACE_RADIUS + self.slope * float(z)


@dataclass(frozen=True, slots=True)
class SwitchSphereLayer:
    """One sphere's certified position relative to the tangent plane."""

    plane_side: int
    item_id: str
    plane_ray_parameter: float

    @property
    def plane_is_in_front(self) -> bool:
        return self.plane_ray_parameter > 0.0


@dataclass(frozen=True, slots=True)
class SwitchOcclusionFrame:
    """Renderer-neutral teaching-transparent painter evidence for one frame."""

    progress: float
    surface_back_item_id: str
    surface_front_item_id: str
    plane_item_ids: tuple[tuple[PlaneDepthRole, str, str], ...]
    plane_contours: tuple[
        tuple[
            PlaneDepthRole,
            tuple[tuple[tuple[float, float], ...], ...],
        ],
        ...,
    ]
    plane_outline_paths: tuple[
        tuple[
            PlaneDepthRole,
            tuple[tuple[tuple[float, float], ...], ...],
        ],
        ...,
    ]
    sphere_layers: tuple[SwitchSphereLayer, SwitchSphereLayer]
    hidden_section_item_id: str
    visible_section_item_id: str
    focus_item_id: str
    draw_order: tuple[str, ...]
    surface_layering_authoritative: bool = True
    physical_surface_visibility_authoritative: bool = False

    def contours_for(
        self,
        role: PlaneDepthRole,
    ) -> tuple[tuple[tuple[float, float], ...], ...]:
        return dict(self.plane_contours)[role]

    def outline_paths_for(
        self,
        role: PlaneDepthRole,
    ) -> tuple[tuple[tuple[float, float], ...], ...]:
        return dict(self.plane_outline_paths)[role]


def compute_switch_frame(progress: object) -> DandelinSwitchFrame:
    """Solve the two tangent spheres for one cone/cylinder family member."""

    resolved = _progress(progress)
    slope = CONE_SLOPE * (1.0 - resolved)
    normalization = sqrt(1.0 + slope * slope)
    radius_intercept = SURFACE_RADIUS / normalization
    radius_gradient = slope / normalization
    normal = np.asarray(PLANE_NORMAL, dtype=float)
    axial_normal = float(normal[2])
    records: list[DandelinSphereFrame] = []

    for plane_side in (-1, 1):
        denominator = axial_normal - plane_side * radius_gradient
        if denominator <= 0.0:
            raise ValueError("cutting plane no longer admits two finite spheres")
        center_z = (
            PLANE_OFFSET + plane_side * radius_intercept
        ) / denominator
        radius = radius_intercept + radius_gradient * center_z
        center = np.asarray((0.0, 0.0, center_z), dtype=float)
        signed_plane_distance = axial_normal * center_z - PLANE_OFFSET
        plane_contact = center - signed_plane_distance * normal

        contact_radius = (
            SURFACE_RADIUS + slope * center_z
        ) / (1.0 + slope * slope)
        contact_z = (
            center_z - slope * SURFACE_RADIUS
        ) / (1.0 + slope * slope)
        if radius <= 0.0 or not all(
            isfinite(item)
            for item in (
                center_z,
                radius,
                contact_radius,
                contact_z,
                *plane_contact,
            )
        ):
            raise ValueError("Dandelin sphere solve produced a non-finite frame")
        records.append(
            DandelinSphereFrame(
                plane_side=plane_side,
                center=(0.0, 0.0, float(center_z)),
                radius=float(radius),
                surface_contact_radius=float(contact_radius),
                surface_contact_z=float(contact_z),
                plane_contact=tuple(float(item) for item in plane_contact),
            )
        )

    apex_z = None if slope == 0.0 else -SURFACE_RADIUS / slope
    return DandelinSwitchFrame(
        progress=resolved,
        slope=float(slope),
        apex_z=(None if apex_z is None else float(apex_z)),
        spheres=(records[0], records[1]),
    )


def _surface_spec(frame: DandelinSwitchFrame) -> ConeSpec | CylinderSpec:
    if frame.slope == 0.0:
        return CylinderSpec(
            "switch-surface",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            SURFACE_RADIUS,
            AXIAL_RANGE,
            radial_axis=(1.0, 0.0, 0.0),
        )
    if frame.apex_z is None:  # pragma: no cover - guarded by slope above
        raise ValueError("a non-cylindrical switch frame requires a finite apex")
    return ConeSpec(
        "switch-surface",
        (0.0, 0.0, frame.apex_z),
        (0.0, 0.0, 1.0),
        atan(frame.slope),
        (
            AXIAL_RANGE[0] - frame.apex_z,
            AXIAL_RANGE[1] - frame.apex_z,
        ),
        radial_axis=(1.0, 0.0, 0.0),
        model=ConeModel.OPEN_SINGLE,
    )


@lru_cache(maxsize=128)
def _compute_switch_occlusion_frame(progress: float) -> SwitchOcclusionFrame:
    geometry = compute_switch_frame(progress)
    surface = _surface_spec(geometry)
    base = compute_global_quadric_frame(
        (),
        (surface,),
        _PARALLEL_VIEW,
        paint_policy=QuadricPaintPolicy.PHYSICAL,
        max_chord_error=_SECTION_MAX_SCREEN_ERROR,
        max_segments=_SECTION_MAX_SEGMENTS,
    ).frame
    section = compute_quadric_section_compositing(
        base,
        surface,
        _SECTION_PLANE,
        _PLANE_PATCH,
        _PARALLEL_VIEW,
        max_screen_error=_SECTION_MAX_SCREEN_ERROR,
    )
    contours = merge_quadric_plane_fragment_contours(
        _SECTION_PLANE,
        _PLANE_PATCH,
        _PARALLEL_VIEW.projection_matrix,
        section.plane_fragments,
    )
    outline_paths: dict[
        PlaneDepthRole,
        list[tuple[tuple[float, float], ...]],
    ] = {role: [] for role in PlaneDepthRole}
    for fragment in section.plane_outline_fragments:
        outline_paths[fragment.role].append(
            (fragment.screen_start, fragment.screen_end)
        )

    paint_items = section.paint_items
    fill_ids = {
        PlaneDepthRole.BEHIND_SURFACE: paint_items.plane_behind,
        PlaneDepthRole.OUTSIDE_PROJECTION: paint_items.plane_outside,
        PlaneDepthRole.BETWEEN_SURFACE_SHEETS: paint_items.plane_between,
        PlaneDepthRole.IN_FRONT_OF_SURFACE: paint_items.plane_front,
    }
    outline_ids = paint_items.outline_by_role
    plane_item_ids = tuple(
        (role, fill_ids[role], outline_ids[role]) for role in PlaneDepthRole
    )

    denominator = float(
        np.dot(
            np.asarray(_SECTION_PLANE.normal, dtype=float),
            np.asarray(_PARALLEL_VIEW.view_direction, dtype=float),
        )
    )
    if abs(denominator) <= 1.0e-12:
        raise ValueError("the cutting plane is edge-on to the fixed view")
    sphere_layers: list[SwitchSphereLayer] = []
    for sphere in geometry.spheres:
        signed_distance = _SECTION_PLANE.signed_distance(sphere.center)
        tangency_error = abs(abs(signed_distance) - sphere.radius)
        if tangency_error > max(1.0e-10, 1.0e-10 * sphere.radius):
            raise ValueError("a switch sphere lost its plane-tangency certificate")
        parameter = -signed_distance / denominator
        if abs(parameter) <= 1.0e-12:
            raise ValueError("a switch sphere has unresolved plane depth")
        sphere_layers.append(
            SwitchSphereLayer(
                sphere.plane_side,
                f"switch-sphere:{sphere.plane_side:+d}",
                float(parameter),
            )
        )
    far_spheres = tuple(
        item for item in sphere_layers if item.plane_is_in_front
    )
    near_spheres = tuple(
        item for item in sphere_layers if not item.plane_is_in_front
    )
    if len(far_spheres) != 1 or len(near_spheres) != 1:
        raise ValueError("the fixed view must separate the two switch spheres")
    far_sphere = far_spheres[0]
    near_sphere = near_spheres[0]

    hidden_section_item_id = "switch-section:hidden"
    visible_section_item_id = "switch-section:visible"
    focus_item_id = "switch-foci"
    base_nodes = tuple(section.draw_order)
    nodes = (
        *base_nodes,
        *(item.item_id for item in sphere_layers),
        hidden_section_item_id,
        visible_section_item_id,
        focus_item_id,
    )
    relations = [
        PainterConstraint(item.far_item_id, item.near_item_id)
        for item in section.order_relations
    ]
    surface_back = paint_items.surface_back
    surface_front = paint_items.surface_front
    behind_nodes = (
        fill_ids[PlaneDepthRole.BEHIND_SURFACE],
        outline_ids[PlaneDepthRole.BEHIND_SURFACE],
        fill_ids[PlaneDepthRole.OUTSIDE_PROJECTION],
        outline_ids[PlaneDepthRole.OUTSIDE_PROJECTION],
        fill_ids[PlaneDepthRole.BETWEEN_SURFACE_SHEETS],
        outline_ids[PlaneDepthRole.BETWEEN_SURFACE_SHEETS],
    )
    front_nodes = (
        fill_ids[PlaneDepthRole.OUTSIDE_PROJECTION],
        outline_ids[PlaneDepthRole.OUTSIDE_PROJECTION],
        fill_ids[PlaneDepthRole.BETWEEN_SURFACE_SHEETS],
        outline_ids[PlaneDepthRole.BETWEEN_SURFACE_SHEETS],
        fill_ids[PlaneDepthRole.IN_FRONT_OF_SURFACE],
        outline_ids[PlaneDepthRole.IN_FRONT_OF_SURFACE],
    )
    plane_front_nodes = (
        fill_ids[PlaneDepthRole.IN_FRONT_OF_SURFACE],
        outline_ids[PlaneDepthRole.IN_FRONT_OF_SURFACE],
    )
    for sphere in sphere_layers:
        relations.extend(
            (
                PainterConstraint(surface_back, sphere.item_id),
                PainterConstraint(sphere.item_id, surface_front),
            )
        )
        if sphere.plane_is_in_front:
            relations.extend(
                PainterConstraint(sphere.item_id, item_id)
                for item_id in front_nodes
            )
            relations.extend(
                PainterConstraint(item_id, sphere.item_id)
                for item_id in (
                    fill_ids[PlaneDepthRole.BEHIND_SURFACE],
                    outline_ids[PlaneDepthRole.BEHIND_SURFACE],
                )
            )
        else:
            relations.extend(
                PainterConstraint(item_id, sphere.item_id)
                for item_id in behind_nodes
            )
            relations.extend(
                PainterConstraint(sphere.item_id, item_id)
                for item_id in plane_front_nodes
            )
    relations.append(
        PainterConstraint(far_sphere.item_id, near_sphere.item_id)
    )
    relations.extend(
        (
            PainterConstraint(surface_back, hidden_section_item_id),
            PainterConstraint(far_sphere.item_id, hidden_section_item_id),
            PainterConstraint(
                fill_ids[PlaneDepthRole.BETWEEN_SURFACE_SHEETS],
                hidden_section_item_id,
            ),
            PainterConstraint(
                outline_ids[PlaneDepthRole.BETWEEN_SURFACE_SHEETS],
                hidden_section_item_id,
            ),
            PainterConstraint(hidden_section_item_id, near_sphere.item_id),
            PainterConstraint(hidden_section_item_id, surface_front),
        )
    )
    core_nodes = (*base_nodes, *(item.item_id for item in sphere_layers))
    relations.extend(
        PainterConstraint(item_id, visible_section_item_id)
        for item_id in (*core_nodes, hidden_section_item_id)
    )
    relations.extend(
        PainterConstraint(item_id, focus_item_id)
        for item_id in (*core_nodes, hidden_section_item_id, visible_section_item_id)
    )

    preferred: list[str] = []
    for item_id in base_nodes:
        preferred.append(item_id)
        if item_id == surface_back:
            preferred.append(far_sphere.item_id)
        if item_id == outline_ids[PlaneDepthRole.BETWEEN_SURFACE_SHEETS]:
            preferred.extend((hidden_section_item_id, near_sphere.item_id))
    preferred.extend((visible_section_item_id, focus_item_id))
    preferred_rank = {
        item_id: index for index, item_id in enumerate(preferred)
    }
    draw_order = stable_topological_sort(
        nodes,
        relations,
        key=lambda item_id: (preferred_rank.get(item_id, len(preferred)), item_id),
    )
    return SwitchOcclusionFrame(
        progress=progress,
        surface_back_item_id=surface_back,
        surface_front_item_id=surface_front,
        plane_item_ids=plane_item_ids,
        plane_contours=tuple(
            (role, contours[role]) for role in PlaneDepthRole
        ),
        plane_outline_paths=tuple(
            (role, tuple(outline_paths[role])) for role in PlaneDepthRole
        ),
        sphere_layers=(sphere_layers[0], sphere_layers[1]),
        hidden_section_item_id=hidden_section_item_id,
        visible_section_item_id=visible_section_item_id,
        focus_item_id=focus_item_id,
        draw_order=draw_order,
    )


def compute_switch_occlusion_frame(progress: object) -> SwitchOcclusionFrame:
    """Certify plane, mother-surface, and sphere order for one progress."""

    return _compute_switch_occlusion_frame(_progress(progress))


def section_point(frame: DandelinSwitchFrame, theta: float) -> tuple[float, float, float]:
    """Return one analytic point on the plane/surface intersection."""

    azimuth_cosine = cos(float(theta))
    denominator = PLANE_NORMAL[2] + (
        PLANE_NORMAL[0] * frame.slope * azimuth_cosine
    )
    if denominator <= 0.0:
        raise ValueError("section parameterization lost its finite branch")
    z = (
        PLANE_OFFSET
        - PLANE_NORMAL[0] * SURFACE_RADIUS * azimuth_cosine
    ) / denominator
    radius = frame.radius_at(z)
    return (
        float(radius * azimuth_cosine),
        float(radius * sin(float(theta))),
        float(z),
    )


def _surface_point(frame: DandelinSwitchFrame, z: float, theta: float) -> np.ndarray:
    radius = frame.radius_at(z)
    return np.asarray(
        (radius * cos(theta), radius * sin(theta), z),
        dtype=float,
    )


def project_point(point: Sequence[float]) -> np.ndarray:
    """Project a world point into the fixed orthographic teaching view."""

    value = np.asarray(tuple(point), dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("projected point must contain three finite coordinates")
    screen = PROJECTION_SCALE * np.asarray(
        (
            float(np.dot(value, _SCREEN_RIGHT)),
            float(np.dot(value, _SCREEN_UP)),
        )
    )
    screen += PROJECTION_OFFSET
    return np.asarray((screen[0], screen[1], 0.0), dtype=float)


def _depth(point: Sequence[float]) -> float:
    return float(np.dot(np.asarray(tuple(point), dtype=float), _DEPTH_AXIS))


def _path(
    points: Sequence[Sequence[float]],
    *,
    color: str,
    width: float,
    opacity: float,
    close: bool = False,
) -> VMobject:
    projected = [project_point(item) for item in points]
    if close and projected:
        projected.append(projected[0])
    result = VMobject()
    if projected:
        result.set_points_as_corners(projected)
    result.set_fill(opacity=0.0)
    result.set_stroke(color=color, width=width, opacity=opacity)
    return result


def _front_runs(
    points: Sequence[Sequence[float]],
    visible: Sequence[bool],
) -> tuple[tuple[Sequence[float], ...], ...]:
    if len(points) != len(visible):
        raise ValueError("visibility flags must match the cyclic point count")
    if not points or not any(visible):
        return ()
    if all(visible):
        return (tuple((*points, points[0])),)

    hidden_index = next(index for index, value in enumerate(visible) if not value)
    ordered = [
        (hidden_index + 1 + offset) % len(points)
        for offset in range(len(points))
    ]
    runs: list[tuple[Sequence[float], ...]] = []
    active: list[Sequence[float]] = []
    for index in ordered:
        if visible[index]:
            active.append(points[index])
        elif active:
            if len(active) >= 2:
                runs.append(tuple(active))
            active = []
    if active and len(active) >= 2:
        runs.append(tuple(active))
    return tuple(runs)


def _tag_paint_item(
    item: VMobject | VGroup,
    item_id: str,
    *,
    kind: str,
    **metadata: object,
) -> VMobject | VGroup:
    payload = {
        "switchPaintItemId": item_id,
        "switchPaintKind": kind,
        **metadata,
    }
    item.switch_paint_item_id = item_id
    item.switch_metadata = payload
    item.metadata = payload
    return item


def _screen_point(point: Sequence[float]) -> np.ndarray:
    value = np.asarray(tuple(point), dtype=float)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise ValueError("screen point must contain two finite coordinates")
    return np.asarray(
        (
            value[0] + PROJECTION_OFFSET[0],
            value[1] + PROJECTION_OFFSET[1],
            0.0,
        ),
        dtype=float,
    )


def _compound_screen_fill(
    paths: Sequence[Sequence[Sequence[float]]],
    *,
    color: str,
    opacity: float,
) -> VMobject:
    value = VMobject()
    for raw_path in paths:
        points = tuple(_screen_point(point) for point in raw_path)
        if len(points) < 3:
            raise ValueError("a plane fill contour requires at least three points")
        value.start_new_path(points[0])
        value.add_points_as_corners((*points[1:], points[0]))
    value.set_fill(color=color, opacity=opacity)
    value.set_stroke(opacity=0.0)
    return value


def _surface_sheet_groups(
    frame: DandelinSwitchFrame,
    back_item_id: str,
    front_item_id: str,
) -> tuple[VGroup, VGroup]:
    z_min, z_max = AXIAL_RANGE
    theta_values = np.linspace(0.0, 2.0 * pi, 25)
    back_patches: list[tuple[float, Polygon]] = []
    front_patches: list[tuple[float, Polygon]] = []
    for index, (theta_0, theta_1) in enumerate(
        zip(theta_values, theta_values[1:])
    ):
        world = (
            _surface_point(frame, z_min, float(theta_0)),
            _surface_point(frame, z_min, float(theta_1)),
            _surface_point(frame, z_max, float(theta_1)),
            _surface_point(frame, z_max, float(theta_0)),
        )
        average_depth = sum(_depth(item) for item in world) / 4.0
        normalized = 0.5 + 0.5 * cos(0.5 * (theta_0 + theta_1) - 0.45)
        color = SURFACE_COLORS[min(3, int(4.0 * normalized))]
        polygon = Polygon(
            *(project_point(item) for item in world),
            stroke_width=0.0,
            fill_color=color,
            fill_opacity=0.11 + 0.10 * normalized,
        )
        theta_mid = 0.5 * (theta_0 + theta_1)
        outward = np.asarray(
            (cos(theta_mid), sin(theta_mid), -frame.slope),
            dtype=float,
        )
        target = (
            front_patches
            if float(np.dot(outward, _DEPTH_AXIS)) > 0.0
            else back_patches
        )
        target.append((average_depth, polygon))
    back_patches.sort(key=lambda item: item[0])
    front_patches.sort(key=lambda item: item[0])
    back = VGroup(*(polygon for _, polygon in back_patches))
    front = VGroup(*(polygon for _, polygon in front_patches))

    for z in AXIAL_RANGE:
        rim = tuple(
            _surface_point(frame, z, float(theta))
            for theta in np.linspace(0.0, 2.0 * pi, 97)[:-1]
        )
        rim_visible = tuple(
            float(
                np.dot(
                    np.asarray(
                        (cos(float(theta)), sin(float(theta)), -frame.slope),
                        dtype=float,
                    ),
                    _DEPTH_AXIS,
                )
            )
            > 0.0
            for theta in np.linspace(0.0, 2.0 * pi, 97)[:-1]
        )
        for run in _front_runs(rim, tuple(not item for item in rim_visible)):
            back.add(
                _path(
                    run,
                    color=SURFACE_STROKE,
                    width=1.15,
                    opacity=0.30,
                )
            )
        for run in _front_runs(rim, rim_visible):
            front.add(
                _path(
                    run,
                    color=SURFACE_STROKE,
                    width=1.55,
                    opacity=0.74,
                )
            )

    radial_view_length = float(np.hypot(_DEPTH_AXIS[0], _DEPTH_AXIS[1]))
    silhouette_ratio = frame.slope * float(_DEPTH_AXIS[2]) / radial_view_length
    if abs(silhouette_ratio) > 1.0 + 1.0e-12:
        raise ValueError("surface silhouette has no finite generator")
    silhouette_ratio = min(1.0, max(-1.0, silhouette_ratio))
    silhouette_phase = atan2(float(_DEPTH_AXIS[1]), float(_DEPTH_AXIS[0]))
    silhouette_offset = acos(silhouette_ratio)
    for theta in (
        silhouette_phase - silhouette_offset,
        silhouette_phase + silhouette_offset,
    ):
        generator = _path(
            (
                _surface_point(frame, z_min, theta),
                _surface_point(frame, z_max, theta),
            ),
            color=SURFACE_STROKE,
            width=1.55,
            opacity=0.78,
        )
        front.add(generator)
    return (
        _tag_paint_item(
            back,
            back_item_id,
            kind="surface_sheet",
            sheetSide="back",
        ),
        _tag_paint_item(
            front,
            front_item_id,
            kind="surface_sheet",
            sheetSide="front",
        ),
    )


def _plane_paint_items(
    occlusion: SwitchOcclusionFrame,
) -> dict[str, VMobject | VGroup]:
    result: dict[str, VMobject | VGroup] = {}
    for role, fill_item_id, outline_item_id in occlusion.plane_item_ids:
        occluded = role in _OCCLUDED_PLANE_ROLES
        fill = _compound_screen_fill(
            occlusion.contours_for(role),
            color=OCCLUDED_PLANE_FILL if occluded else PLANE_COLOR,
            opacity=0.18 if occluded else 0.10,
        )
        result[fill_item_id] = _tag_paint_item(
            fill,
            fill_item_id,
            kind="plane_fill",
            planeDepthRole=role.value,
            planeOccludedBySurface=occluded,
        )

        outline_members: list[VMobject | DashedVMobject] = []
        for path in occlusion.outline_paths_for(role):
            base = VMobject()
            points = tuple(_screen_point(point) for point in path)
            base.set_points_as_corners(points)
            base.set_fill(opacity=0.0)
            base.set_stroke(
                color=OCCLUDED_PLANE_STROKE if occluded else "#86F7E4",
                width=1.0 if occluded else 1.2,
                opacity=0.52 if occluded else 0.62,
            )
            if occluded:
                length = float(base.get_arc_length())
                if length <= 0.0:
                    continue
                dashed = DashedVMobject(
                    base,
                    num_dashes=max(2, min(96, int(ceil(length / 0.24)))),
                    dashed_ratio=0.52,
                )
                dashed.set_stroke(
                    color=OCCLUDED_PLANE_STROKE,
                    width=1.0,
                    opacity=0.52,
                )
                outline_members.append(dashed)
            else:
                outline_members.append(base)
        outline = VGroup(*outline_members)
        result[outline_item_id] = _tag_paint_item(
            outline,
            outline_item_id,
            kind="plane_outline",
            planeDepthRole=role.value,
            planeOccludedBySurface=occluded,
            strokePattern="dashed" if occluded else "solid",
        )
    return result


def _sphere_paint_items(
    frame: DandelinSwitchFrame,
    occlusion: SwitchOcclusionFrame,
) -> tuple[dict[str, VGroup], VGroup]:
    item_id_by_side = {
        layer.plane_side: layer.item_id for layer in occlusion.sphere_layers
    }
    items: dict[str, VGroup] = {}
    foci = VGroup()
    for sphere in frame.spheres:
        members = VGroup()
        center = project_point(sphere.center)
        screen_radius = PROJECTION_SCALE * sphere.radius
        body = Circle(
            radius=screen_radius,
            fill_color=SPHERE_COLOR,
            fill_opacity=0.25,
            stroke_color=SPHERE_STROKE,
            stroke_width=1.7,
            stroke_opacity=0.82,
        ).move_to(center)
        glow = Circle(
            radius=0.70 * screen_radius,
            fill_color="#FFC2A6",
            fill_opacity=0.07,
            stroke_opacity=0.0,
        ).move_to(center + np.asarray((-0.12, 0.16, 0.0)) * screen_radius)
        members.add(body, glow)

        ring_points = tuple(
            (
                sphere.surface_contact_radius * cos(float(theta)),
                sphere.surface_contact_radius * sin(float(theta)),
                sphere.surface_contact_z,
            )
            for theta in np.linspace(0.0, 2.0 * pi, 97)[:-1]
        )
        ring_visible = tuple(
            float(
                np.dot(
                    np.asarray((cos(float(theta)), sin(float(theta)), -frame.slope)),
                    _DEPTH_AXIS,
                )
            )
            > 0.0
            for theta in np.linspace(0.0, 2.0 * pi, 97)[:-1]
        )
        hidden_ring = _path(
            ring_points,
            color=SPHERE_COLOR,
            width=1.5,
            opacity=0.30,
            close=True,
        )
        members.add(hidden_ring)
        for run in _front_runs(ring_points, ring_visible):
            visible_ring = _path(
                run,
                color=SPHERE_COLOR,
                width=3.0,
                opacity=1.0,
            )
            members.add(visible_ring)

        focus = project_point(sphere.plane_contact)
        halo = Dot(focus, radius=0.10, color=FOCUS_COLOR, fill_opacity=0.18)
        point = Dot(focus, radius=0.048, color=FOCUS_COLOR)
        foci.add(halo, point)
        item_id = item_id_by_side[sphere.plane_side]
        items[item_id] = _tag_paint_item(
            members,
            item_id,
            kind="sphere",
            planeSide=sphere.plane_side,
        )
    return items, _tag_paint_item(
        foci,
        occlusion.focus_item_id,
        kind="focus",
    )


def _section_paint_items(
    frame: DandelinSwitchFrame,
    occlusion: SwitchOcclusionFrame,
) -> tuple[VMobject, VGroup]:
    theta_values = np.linspace(0.0, 2.0 * pi, 145)[:-1]
    points = tuple(section_point(frame, float(theta)) for theta in theta_values)
    near_layer = next(
        layer for layer in occlusion.sphere_layers if not layer.plane_is_in_front
    )
    near_sphere = next(
        sphere
        for sphere in frame.spheres
        if sphere.plane_side == near_layer.plane_side
    )
    near_center = project_point(near_sphere.center)
    near_radius = PROJECTION_SCALE * near_sphere.radius
    visible: list[bool] = []
    for theta, point in zip(theta_values, points):
        surface_facing = float(
            np.dot(
                np.asarray((cos(float(theta)), sin(float(theta)), -frame.slope)),
                _DEPTH_AXIS,
            )
        ) > 0.0
        projected = project_point(point)
        outside_near_sphere = (
            float(np.linalg.norm(projected[:2] - near_center[:2]))
            >= near_radius + 0.018
        )
        visible.append(surface_facing and outside_near_sphere)
    hidden = _path(
        points,
        color=SECTION_COLOR,
        width=1.9,
        opacity=0.30,
        close=True,
    )
    visible_group = VGroup()
    for run in _front_runs(points, visible):
        front = _path(
            run,
            color=SECTION_COLOR,
            width=4.0,
            opacity=1.0,
        )
        visible_group.add(front)
    return (
        _tag_paint_item(
            hidden,
            occlusion.hidden_section_item_id,
            kind="section_hidden",
        ),
        _tag_paint_item(
            visible_group,
            occlusion.visible_section_item_id,
            kind="section_visible",
            sphereOcclusionAware=True,
        ),
    )


def build_switch_diagram(progress: object) -> VGroup:
    """Build one deterministic visual frame for playback or test capture."""

    frame = compute_switch_frame(progress)
    occlusion = compute_switch_occlusion_frame(frame.progress)
    axis = DashedLine(
        project_point((0.0, 0.0, AXIAL_RANGE[0] - 0.15)),
        project_point((0.0, 0.0, AXIAL_RANGE[1] + 0.15)),
        dash_length=0.10,
        color="#A8BED0",
        stroke_width=1.15,
        stroke_opacity=0.30,
    )
    axis.set_z_index(1.0)

    surface_back, surface_front = _surface_sheet_groups(
        frame,
        occlusion.surface_back_item_id,
        occlusion.surface_front_item_id,
    )
    paint_items: dict[str, VMobject | VGroup] = {
        occlusion.surface_back_item_id: surface_back,
        occlusion.surface_front_item_id: surface_front,
        **_plane_paint_items(occlusion),
    }
    spheres, foci = _sphere_paint_items(frame, occlusion)
    paint_items.update(spheres)
    hidden_section, visible_section = _section_paint_items(frame, occlusion)
    paint_items[occlusion.hidden_section_item_id] = hidden_section
    paint_items[occlusion.visible_section_item_id] = visible_section
    paint_items[occlusion.focus_item_id] = foci
    if set(paint_items) != set(occlusion.draw_order):
        missing = sorted(set(occlusion.draw_order) - set(paint_items))
        unexpected = sorted(set(paint_items) - set(occlusion.draw_order))
        raise ValueError(
            "switch painter items disagree with certified draw order: "
            f"missing={missing}, unexpected={unexpected}"
        )
    ordered: list[VMobject | VGroup] = []
    for rank, item_id in enumerate(occlusion.draw_order):
        item = paint_items[item_id]
        item.set_z_index(10.0 + rank, family=True)
        ordered.append(item)
    result = VGroup(axis, *ordered)
    result.switch_occlusion_frame = occlusion
    return result


def _header() -> VGroup:
    title = Text(
        "丹德林双球：圆锥面  ⇄  圆柱面",
        font_size=38,
        color="#F3F7FA",
        weight="SEMIBOLD",
    ).to_edge(UP, buff=0.18)
    subtitle = Text(
        "两只球始终同时与母面、截平面相切",
        font_size=20,
        color="#BFD5E4",
    ).next_to(title, DOWN, buff=0.09)
    group = VGroup(title, subtitle)
    group.set_z_index(100.0)
    return group


def _progress_legend(tracker: ValueTracker) -> VGroup:
    left = Text("圆锥面", font_size=19, color="#77E3F2")
    right = Text("圆柱面", font_size=19, color="#77E3F2")
    track = Line((-2.05, -3.47, 0.0), (2.05, -3.47, 0.0))
    track.set_stroke(color="#5B7184", width=2.0, opacity=0.65)
    left.next_to(track, DOWN, buff=0.13).align_to(track, np.array((-1.0, 0.0, 0.0)))
    right.next_to(track, DOWN, buff=0.13).align_to(track, np.array((1.0, 0.0, 0.0)))
    marker = Dot(track.get_start(), radius=0.075, color=SECTION_COLOR)

    def follow_progress(item: Dot) -> None:
        item.move_to(
            track.get_start()
            + tracker.get_value() * (track.get_end() - track.get_start())
        )

    marker.add_updater(follow_progress)
    group = VGroup(track, marker, left, right)
    group.set_z_index(100.0)
    return group


class DandelinConeCylinderSwitch(Scene):
    """Round-trip cone/cylinder switch with two analytic tangent spheres."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND_COLOR
        progress = ValueTracker(0.0)
        header = _header()
        legend = _progress_legend(progress)
        diagram = always_redraw(lambda: build_switch_diagram(progress.get_value()))

        self.add_foreground_mobjects(header, legend)
        self.play(FadeIn(diagram), FadeIn(header), FadeIn(legend), run_time=0.65)
        self.wait(0.65)
        self.play(
            progress.animate.set_value(1.0),
            run_time=3.8,
            rate_func=smooth,
        )
        self.wait(0.85)
        self.play(
            progress.animate.set_value(0.0),
            run_time=3.8,
            rate_func=smooth,
        )
        self.wait(0.65)


__all__: Sequence[str] = (
    "AXIAL_RANGE",
    "BACKGROUND_COLOR",
    "CONE_SLOPE",
    "DandelinConeCylinderSwitch",
    "DandelinSphereFrame",
    "DandelinSwitchFrame",
    "SwitchOcclusionFrame",
    "SwitchSphereLayer",
    "PLANE_NORMAL",
    "PLANE_OFFSET",
    "PlaneDepthRole",
    "SURFACE_RADIUS",
    "build_switch_diagram",
    "compute_switch_occlusion_frame",
    "compute_switch_frame",
    "project_point",
    "section_point",
)


config.background_color = BACKGROUND_COLOR
