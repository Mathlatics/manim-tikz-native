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

from dataclasses import dataclass, replace
from functools import lru_cache
from math import atan, cos, isfinite, pi, sin, sqrt
from typing import Sequence

import numpy as np
from manim import (
    DOWN,
    UP,
    Circle,
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
    CylinderModel,
    CylinderSpec,
    PlaneDisplayPatchSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.curves import EllipseArcCurve, SegmentCurve
from polyhedron_visibility.quadrics.nested_tangent_compositing import (
    NestedTangentSphereSpec,
)
from polyhedron_visibility.quadrics.manim_runtime import (
    _adaptive_project_curve_samples,
    _dash_polyline_anchored,
    _polyline_lengths,
    _slice_projected_curve_samples,
    _source_distance_at_parameter,
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
    build_surface_boundary_sources,
    curve_boundary_source,
    plane_outline_sources,
    section_curve_boundary_source,
)


BACKGROUND_COLOR = "#0B1622"
GEOMETRY_Z_BASE = 10.0
TEACHING_UI_Z_INDEX = 10_000.0
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
            model=CylinderModel.OPEN,
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
    sources.extend(
        replace(
            source,
            style_id=(
                "style:switch-sphere-silhouette"
                if source.owner_surface_id is not None
                and source.owner_surface_id.startswith("switch-sphere:")
                else "style:switch-surface-boundary"
            ),
        )
        for source in build_surface_boundary_sources(
            (surface, *spheres),
            _PARALLEL_VIEW,
            include_cap_rims=True,
            include_silhouettes=True,
        )
    )
    sources.extend(
        replace(source, style_id="style:switch-plane-outline")
        for source in plane_outline_sources(
            _SECTION_PLANE,
            _PLANE_PATCH,
            occlusion_scope=BoundaryOcclusionScope.ALL_SURFACES,
        )
    )
    axis_id = "switch-axis"
    sources.append(
        curve_boundary_source(
            SegmentCurve(
                axis_id,
                (0.0, 0.0, AXIAL_RANGE[0] - 0.15),
                (0.0, 0.0, AXIAL_RANGE[1] + 0.15),
            ),
            source_kind=BoundarySourceKind.FEATURE_LINE,
            semantic_kind=BoundarySemanticKind.TEACHING_FEATURE,
            occlusion_scope=BoundaryOcclusionScope.ALL_SURFACES,
            owner_id=axis_id,
            style_id="style:switch-axis",
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
    return (
        _tag_paint_item(
            back,
            back_item_id,
            kind="surface_sheet",
            sheetSide="back",
            boundaryFragmentsEmbedded=False,
        ),
        _tag_paint_item(
            front,
            front_item_id,
            kind="surface_sheet",
            sheetSide="front",
            boundaryFragmentsEmbedded=False,
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
            boundaryFragmentsEmbedded=False,
        )

        outline = VGroup()
        result[outline_item_id] = _tag_paint_item(
            outline,
            outline_item_id,
            kind="plane_outline",
            planeDepthRole=role.value,
            planeOccludedBySurface=occluded,
            strokePattern="structural-anchor",
            boundaryFragmentsEmbedded=False,
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
            stroke_width=0.0,
            stroke_opacity=0.0,
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
            silhouetteEmbedded=False,
        )
    return items, _tag_paint_item(
        foci,
        occlusion.focus_item_id,
        kind="focus",
    )


def _projected_path(
    points: Sequence[Sequence[float]],
    *,
    color: str,
    width: float,
    opacity: float,
) -> VMobject:
    result = VMobject()
    values = tuple(np.asarray(point, dtype=float) for point in points)
    if values:
        result.set_points_as_corners(values)
    result.set_fill(opacity=0.0)
    result.set_stroke(color=color, width=width, opacity=opacity)
    return result


def _boundary_paint_items(
    occlusion: SwitchOcclusionFrame,
) -> dict[str, VMobject | VGroup]:
    styles = {
        "style:switch-contact": (
            "contact_curve",
            SPHERE_COLOR,
            SPHERE_COLOR,
            3.0,
            1.5,
            1.0,
            0.30,
            0.18,
            False,
        ),
        "style:switch-section": (
            "section_curve",
            SECTION_COLOR,
            SECTION_COLOR,
            4.0,
            1.9,
            1.0,
            0.30,
            0.18,
            False,
        ),
        "style:switch-sphere-silhouette": (
            "sphere_silhouette",
            SPHERE_STROKE,
            SPHERE_STROKE,
            1.7,
            1.2,
            0.82,
            0.38,
            0.16,
            False,
        ),
        "style:switch-surface-boundary": (
            "surface_boundary",
            SURFACE_STROKE,
            SURFACE_STROKE,
            1.55,
            1.1,
            0.78,
            0.34,
            0.18,
            False,
        ),
        "style:switch-plane-outline": (
            "plane_outline",
            "#86F7E4",
            OCCLUDED_PLANE_STROKE,
            1.2,
            1.0,
            0.62,
            0.52,
            0.24,
            False,
        ),
        "style:switch-axis": (
            "axis",
            "#A8BED0",
            "#A8BED0",
            1.15,
            0.85,
            0.30,
            0.16,
            0.20,
            True,
        ),
    }
    source_map = {item.source_id: item for item in occlusion.curve_sources}
    fragments_by_source = {
        source_id: tuple(
            item
            for item in occlusion.curve_fragments
            if item.source_id == source_id and item.painted
        )
        for source_id in source_map
    }
    projected_sources: dict[
        str,
        tuple[np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    display_offset = np.asarray(
        (PROJECTION_OFFSET[0], PROJECTION_OFFSET[1], 0.0),
        dtype=float,
    )
    for source_id, source in source_map.items():
        fragments = fragments_by_source[source_id]
        required_parameters = tuple(
            value
            for fragment in fragments
            for value in (fragment.interval.start, fragment.interval.end)
        )
        parameters, points = _adaptive_project_curve_samples(
            source.curve,
            _PARALLEL_VIEW,
            required_parameters=required_parameters,
            max_chord_error=0.0025,
            max_segments=512,
        )
        points = points + display_offset
        cumulative, _length = _polyline_lengths(points)
        projected_sources[source_id] = (parameters, points, cumulative)

    result: dict[str, VMobject | VGroup] = {}
    for fragment in occlusion.curve_fragments:
        if not fragment.painted:
            continue
        source = source_map[fragment.source_id]
        visible = (
            fragment.effective_visibility_kind is VisibilityKind.VISIBLE
        )
        try:
            (
                paint_kind,
                visible_color,
                hidden_color,
                visible_width,
                hidden_width,
                visible_opacity,
                hidden_opacity,
                dash_step,
                force_dashed,
            ) = styles[source.style_id]
        except KeyError as exc:
            raise ValueError(
                f"unsupported switch boundary style {source.style_id!r}"
            ) from exc
        color = visible_color if visible else hidden_color
        width = visible_width if visible else hidden_width
        opacity = visible_opacity if visible else hidden_opacity
        parameters, source_points, source_cumulative = projected_sources[
            source.source_id
        ]
        points = _slice_projected_curve_samples(
            parameters,
            source_points,
            fragment.interval.start,
            fragment.interval.end,
            curve_id=source.curve.curve_id,
        )
        base = _projected_path(
            points,
            color=color,
            width=width,
            opacity=opacity,
        )
        value: VMobject | VGroup = base
        dashed = force_dashed or fragment.render_intent is BoundaryRenderIntent.DASHED
        if dashed:
            dashes = _dash_polyline_anchored(
                points,
                source_distance_start=_source_distance_at_parameter(
                    parameters,
                    source_points,
                    fragment.interval.start,
                    cumulative=source_cumulative,
                ),
                dash_length=0.54 * dash_step,
                dash_gap=0.46 * dash_step,
                capacity=256,
            )
            value = VGroup(
                *(
                    _projected_path(
                        dash.points,
                        color=color,
                        width=width,
                        opacity=opacity,
                    )
                    for dash in dashes
                )
            )
        result[fragment.item_id] = _tag_paint_item(
            value,
            fragment.item_id,
            kind=paint_kind,
            sourceId=fragment.source_id,
            sourceKind=source.source_kind.value,
            visibility=fragment.effective_visibility_kind.value,
            strokePattern="dashed" if dashed else "solid",
            dashPhaseAnchored=dashed,
            dashPeriod=dash_step if dashed else 0.0,
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
    if GEOMETRY_Z_BASE + len(occlusion.draw_order) - 1 >= TEACHING_UI_Z_INDEX:
        raise ValueError("switch geometry exhausted the reserved teaching UI layer")
    ordered: list[VMobject | VGroup] = []
    for rank, item_id in enumerate(occlusion.draw_order):
        item = paint_items[item_id]
        item.set_z_index(GEOMETRY_Z_BASE + rank, family=True)
        ordered.append(item)
    result = VGroup(*ordered)
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
    group.set_z_index(TEACHING_UI_Z_INDEX, family=True)
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
    group.set_z_index(TEACHING_UI_Z_INDEX, family=True)
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
