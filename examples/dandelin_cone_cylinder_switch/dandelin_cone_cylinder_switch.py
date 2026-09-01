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
    config,
    smooth,
)

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.visibility import VisibilityKind
from polyhedron_visibility.quadrics.boundary_compositing import (
    BoundaryOcclusionScope,
    BoundaryRenderIntent,
    BoundarySemanticKind,
    BoundarySourceKind,
    QuadricBoundaryPaintFragment,
    QuadricBoundarySource,
)
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    CylinderSpec,
    PlaneDisplayPatchSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.curves import EllipseArcCurve
from polyhedron_visibility.quadrics.nested_tangent_compositing import (
    NestedTangentSphereSpec,
)
from polyhedron_visibility.quadrics.scene_occlusion import (
    SceneOcclusionFrame,
    SceneOcclusionRequest,
    SceneSectionSpec,
    compute_scene_occlusion_frame as compute_registered_scene_occlusion_frame,
)
from polyhedron_visibility.quadrics.section_compositing import (
    PlaneDepthRole,
    merge_quadric_plane_fragment_contours,
)
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section_boundary_curves,
)
from polyhedron_visibility.quadrics.surface_boundaries import (
    curve_boundary_source,
    section_curve_boundary_source,
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
    scene_frame: SceneOcclusionFrame
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

    @property
    def curve_sources(self) -> tuple[QuadricBoundarySource, ...]:
        boundary = self.scene_frame.boundary_frame
        return () if boundary is None else boundary.sources

    @property
    def curve_fragments(self) -> tuple[QuadricBoundaryPaintFragment, ...]:
        boundary = self.scene_frame.boundary_frame
        return () if boundary is None else boundary.fragments

    def __deepcopy__(self, memo: dict[int, object]) -> "SwitchOcclusionFrame":
        """Share this immutable evidence when Manim copies a diagram."""

        memo[id(self)] = self
        return self


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
    spheres: list[SphereSpec] = []
    sources: list[QuadricBoundarySource] = []
    bindings: list[NestedTangentSphereSpec] = []
    for sphere in geometry.spheres:
        sphere_id = f"switch-sphere:{sphere.plane_side:+d}"
        contact_id = f"switch-contact:{sphere.plane_side:+d}"
        sphere_spec = SphereSpec(sphere_id, sphere.center, sphere.radius)
        contact_curve = EllipseArcCurve(
            contact_id,
            (0.0, 0.0, sphere.surface_contact_z),
            (sphere.surface_contact_radius, 0.0, 0.0),
            (0.0, sphere.surface_contact_radius, 0.0),
        )
        spheres.append(sphere_spec)
        sources.append(
            curve_boundary_source(
                contact_curve,
                source_kind=BoundarySourceKind.ANALYTIC_CURVE,
                semantic_kind=BoundarySemanticKind.TEACHING_FEATURE,
                occlusion_scope=BoundaryOcclusionScope.OWNER_AND_EXTERNAL,
                owner_id=sphere_id,
                owner_surface_id=sphere_id,
                style_id="style:switch-contact",
            )
        )
        bindings.append(
            NestedTangentSphereSpec(
                sphere_id,
                surface.surface_id,
                contact_id,
                sphere_id,
            )
        )
    section_curves = compute_quadric_section_boundary_curves(
        "switch-section",
        surface,
        _SECTION_PLANE,
    )
    sources.extend(
        section_curve_boundary_source(
            curve,
            surface,
            _SECTION_PLANE,
            section_id="switch-section",
            authoritative_curves=section_curves,
            style_id="style:switch-section",
        )
        for curve in section_curves
    )
    scene_frame = compute_registered_scene_occlusion_frame(
        SceneOcclusionRequest(
            "dandelin-cone-cylinder-switch",
            (surface, *spheres),
            _PARALLEL_VIEW,
            tuple(sources),
            SceneSectionSpec(surface.surface_id, _SECTION_PLANE, _PLANE_PATCH),
            tuple(bindings),
            QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            max_chord_error=_SECTION_MAX_SCREEN_ERROR,
            max_surface_segments=_SECTION_MAX_SEGMENTS,
        )
    )
    section = scene_frame.section_frame
    nested = scene_frame.nested_parent_frame
    if section is None or nested is None or scene_frame.boundary_frame is None:
        raise ValueError("the switch scene lost its registered occlusion evidence")
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
    contacts = {
        item.sphere_surface_id: item for item in nested.contacts
    }
    sphere_layers = tuple(
        SwitchSphereLayer(
            sphere.plane_side,
            f"switch-sphere:{sphere.plane_side:+d}",
            contacts[f"switch-sphere:{sphere.plane_side:+d}"].plane_ray_parameter,
        )
        for sphere in geometry.spheres
    )
    focus_item_id = "switch-foci"
    return SwitchOcclusionFrame(
        progress=progress,
        scene_frame=scene_frame,
        surface_back_item_id=paint_items.surface_back,
        surface_front_item_id=paint_items.surface_front,
        plane_item_ids=plane_item_ids,
        plane_contours=tuple(
            (role, contours[role]) for role in PlaneDepthRole
        ),
        plane_outline_paths=tuple(
            (role, tuple(outline_paths[role])) for role in PlaneDepthRole
        ),
        sphere_layers=(sphere_layers[0], sphere_layers[1]),
        focus_item_id=focus_item_id,
        draw_order=(*scene_frame.draw_order, focus_item_id),
        surface_layering_authoritative=(
            scene_frame.surface_layering_authoritative
        ),
        physical_surface_visibility_authoritative=(
            scene_frame.physical_surface_visibility_authoritative
        ),
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

        focus = project_point(sphere.plane_contact)
        halo = Dot(focus, radius=0.10, color=FOCUS_COLOR, fill_opacity=0.18)
        point = Dot(focus, radius=0.048, color=FOCUS_COLOR)
        foci.add(halo, point)
        item_id = item_id_by_side[sphere.plane_side]
        items[item_id] = _tag_paint_item(
            members,
            item_id,
            kind="sphere_body",
            planeSide=sphere.plane_side,
            contactCurvesEmbedded=False,
        )
    return items, _tag_paint_item(
        foci,
        occlusion.focus_item_id,
        kind="focus",
    )


def _point_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    delta = end - start
    squared = float(np.dot(delta, delta))
    if squared == 0.0:
        return float(np.linalg.norm(point - start))
    ratio = float(np.dot(point - start, delta) / squared)
    ratio = min(1.0, max(0.0, ratio))
    return float(np.linalg.norm(point - (start + ratio * delta)))


def _fragment_world_points(
    source: QuadricBoundarySource,
    fragment: QuadricBoundaryPaintFragment,
    *,
    max_screen_error: float = 0.0025,
    max_segments: int = 512,
) -> tuple[tuple[float, float, float], ...]:
    """Sample an already-certified interval only for renderer approximation."""

    curve = source.curve
    cache: dict[float, np.ndarray] = {}

    def screen(parameter: float) -> np.ndarray:
        if parameter not in cache:
            cache[parameter] = project_point(curve.point(parameter))[:2]
        return cache[parameter]

    intervals = [(fragment.interval.start, fragment.interval.end)]
    probes = (0.25, 0.5, 0.75)
    while True:
        split_indices: list[int] = []
        for index, (start, end) in enumerate(intervals):
            first = screen(start)
            last = screen(end)
            observed = max(
                _point_segment_distance(
                    screen(start + fraction * (end - start)),
                    first,
                    last,
                )
                for fraction in probes
            )
            if observed > max_screen_error:
                split_indices.append(index)
        if not split_indices:
            break
        if len(intervals) + len(split_indices) > max_segments:
            raise ValueError(
                f"curve fragment {fragment.item_id!r} exceeds display capacity"
            )
        split_set = set(split_indices)
        refined: list[tuple[float, float]] = []
        for index, (start, end) in enumerate(intervals):
            if index not in split_set:
                refined.append((start, end))
                continue
            midpoint = 0.5 * (start + end)
            refined.extend(((start, midpoint), (midpoint, end)))
        intervals = refined
    parameters = (intervals[0][0], *(end for _start, end in intervals))
    return tuple(curve.point(parameter) for parameter in parameters)


def _boundary_paint_items(
    occlusion: SwitchOcclusionFrame,
) -> dict[str, VMobject | DashedVMobject]:
    source_map = {item.source_id: item for item in occlusion.curve_sources}
    result: dict[str, VMobject | DashedVMobject] = {}
    for fragment in occlusion.curve_fragments:
        if not fragment.painted:
            continue
        source = source_map[fragment.source_id]
        contact = source.style_id == "style:switch-contact"
        visible = (
            fragment.effective_visibility_kind is VisibilityKind.VISIBLE
        )
        base = _path(
            _fragment_world_points(source, fragment),
            color=SPHERE_COLOR if contact else SECTION_COLOR,
            width=(3.0 if visible else 1.5) if contact else (4.0 if visible else 1.9),
            opacity=1.0 if visible else 0.30,
        )
        value: VMobject | DashedVMobject = base
        if fragment.render_intent is BoundaryRenderIntent.DASHED:
            length = float(base.get_arc_length())
            if length > 0.0:
                value = DashedVMobject(
                    base,
                    num_dashes=max(2, min(96, int(ceil(length / 0.18)))),
                    dashed_ratio=0.54,
                )
                value.set_stroke(
                    color=SPHERE_COLOR if contact else SECTION_COLOR,
                    width=1.5 if contact else 1.9,
                    opacity=0.30,
                )
        result[fragment.item_id] = _tag_paint_item(
            value,
            fragment.item_id,
            kind="contact_curve" if contact else "section_curve",
            sourceId=fragment.source_id,
            visibility=fragment.effective_visibility_kind.value,
            occluderSurfaceIds=fragment.occluder_surface_ids,
            analyticInterval=(
                fragment.interval.start,
                fragment.interval.end,
            ),
        )
    return result


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
    paint_items.update(_boundary_paint_items(occlusion))
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


def refresh_switch_diagram(diagram: VGroup, progress: object) -> VGroup:
    """Atomically replace one live diagram with a newly certified frame.

    ``always_redraw`` delegates to ``Mobject.become``.  That operation aligns
    unequal submobject trees and intentionally retains the old allocation,
    which is unsafe here because analytic curve events can change the number
    of boundary fragments.  The outer diagram keeps its Scene identity while
    this function swaps its complete, already-ordered child list as one unit.
    """

    if not isinstance(diagram, VGroup):
        raise TypeError("diagram must be a VGroup")
    replacement = build_switch_diagram(progress)
    diagram.submobjects = list(replacement.submobjects)
    diagram.switch_occlusion_frame = replacement.switch_occlusion_frame
    return diagram


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

    def get_moving_and_static_mobjects(self, animations):
        """Keep Cairo's moving set rooted above dynamic analytic fragments.

        Boundary events can split or merge painter fragments while the switch
        is running.  Cairo normally freezes a flattened family list at the
        beginning of each animation, so replacing those children would leave
        stale family members in its frame cache.  Returning the stable scene
        roots makes Cairo expand the *current* family on every frame.  This is
        deliberately local to the example; the renderer-neutral occlusion
        coordinator still returns immutable frame evidence.
        """

        del animations
        roots: list[VMobject] = []
        for item in (*self.mobjects, *self.foreground_mobjects):
            if not any(item is existing for existing in roots):
                roots.append(item)
        return roots, []

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND_COLOR
        progress = ValueTracker(0.0)
        header = _header()
        legend = _progress_legend(progress)
        diagram = build_switch_diagram(progress.get_value())

        self.add_foreground_mobjects(header, legend)
        self.play(FadeIn(diagram), FadeIn(header), FadeIn(legend), run_time=0.65)
        diagram.add_updater(
            lambda item: refresh_switch_diagram(item, progress.get_value())
        )
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
    "refresh_switch_diagram",
    "section_point",
)


config.background_color = BACKGROUND_COLOR
