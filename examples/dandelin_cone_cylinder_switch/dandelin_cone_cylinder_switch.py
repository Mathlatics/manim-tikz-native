"""Animate an analytic Dandelin two-sphere cone-to-cylinder limit.

The displayed surface family is

``r(z, p) = R + k(p) z`` with ``k(p) = k0 (1 - p)``.

At ``p = 0`` the lower trim circle collapses to the cone apex.  As ``p``
approaches one, the apex recedes to negative infinity and the generators
become parallel.  At ``p = 1`` the family is exactly a cylinder.  Both sphere
centres, radii, surface-contact circles, and plane-contact points are solved
analytically for every frame.

Preview with::

    manim --renderer cairo --disable_caching -ql --fps 12 \
      examples/dandelin_cone_cylinder_switch/dandelin_cone_cylinder_switch.py \
      DandelinConeCylinderSwitch
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sin, sqrt
from typing import Sequence

import numpy as np
from manim import (
    DOWN,
    UP,
    Circle,
    DashedLine,
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


def _surface_group(frame: DandelinSwitchFrame) -> VGroup:
    z_min, z_max = AXIAL_RANGE
    theta_values = np.linspace(0.0, 2.0 * pi, 25)
    patches: list[tuple[float, Polygon]] = []
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
        patches.append((average_depth, polygon))
    patches.sort(key=lambda item: item[0])
    group = VGroup()
    for index, (_, polygon) in enumerate(patches):
        polygon.set_z_index(10.0 + index * 0.001)
        group.add(polygon)

    for z in AXIAL_RANGE:
        rim = tuple(
            _surface_point(frame, z, float(theta))
            for theta in np.linspace(0.0, 2.0 * pi, 97)[:-1]
        )
        line = _path(
            rim,
            color=SURFACE_STROKE,
            width=1.45,
            opacity=0.64,
            close=True,
        )
        line.set_z_index(12.0)
        group.add(line)

    silhouette_theta = float(np.arctan2(_SCREEN_RIGHT[1], _SCREEN_RIGHT[0]))
    for theta in (silhouette_theta, silhouette_theta + pi):
        generator = _path(
            (
                _surface_point(frame, z_min, theta),
                _surface_point(frame, z_max, theta),
            ),
            color=SURFACE_STROKE,
            width=1.55,
            opacity=0.78,
        )
        generator.set_z_index(12.1)
        group.add(generator)
    return group


def _plane_group() -> VGroup:
    normal = np.asarray(PLANE_NORMAL, dtype=float)
    center = PLANE_OFFSET * normal
    u_axis = np.asarray((0.0, 1.0, 0.0), dtype=float)
    v_axis = np.asarray((normal[2], 0.0, -normal[0]), dtype=float)
    corners = tuple(
        center + u_sign * 3.35 * u_axis + v_sign * 3.55 * v_axis
        for u_sign, v_sign in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    )
    patch = Polygon(
        *(project_point(item) for item in corners),
        fill_color=PLANE_COLOR,
        fill_opacity=0.10,
        stroke_color="#86F7E4",
        stroke_width=1.2,
        stroke_opacity=0.56,
    )
    patch.set_z_index(5.0)
    return VGroup(patch)


def _sphere_group(frame: DandelinSwitchFrame) -> VGroup:
    group = VGroup()
    for sphere in frame.spheres:
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
        body.set_z_index(20.0)
        glow = Circle(
            radius=0.70 * screen_radius,
            fill_color="#FFC2A6",
            fill_opacity=0.07,
            stroke_opacity=0.0,
        ).move_to(center + np.asarray((-0.12, 0.16, 0.0)) * screen_radius)
        glow.set_z_index(20.1)
        group.add(body, glow)

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
        hidden_ring.set_z_index(21.0)
        group.add(hidden_ring)
        for run in _front_runs(ring_points, ring_visible):
            visible_ring = _path(
                run,
                color=SPHERE_COLOR,
                width=3.0,
                opacity=1.0,
            )
            visible_ring.set_z_index(21.1)
            group.add(visible_ring)

        focus = project_point(sphere.plane_contact)
        halo = Dot(focus, radius=0.10, color=FOCUS_COLOR, fill_opacity=0.18)
        point = Dot(focus, radius=0.048, color=FOCUS_COLOR)
        halo.set_z_index(23.0)
        point.set_z_index(23.1)
        group.add(halo, point)
    return group


def _section_group(frame: DandelinSwitchFrame) -> VGroup:
    theta_values = np.linspace(0.0, 2.0 * pi, 145)[:-1]
    points = tuple(section_point(frame, float(theta)) for theta in theta_values)
    visible = tuple(
        float(
            np.dot(
                np.asarray((cos(float(theta)), sin(float(theta)), -frame.slope)),
                _DEPTH_AXIS,
            )
        )
        > 0.0
        for theta in theta_values
    )
    group = VGroup()
    hidden = _path(
        points,
        color=SECTION_COLOR,
        width=2.1,
        opacity=0.34,
        close=True,
    )
    hidden.set_z_index(24.0)
    group.add(hidden)
    for run in _front_runs(points, visible):
        front = _path(
            run,
            color=SECTION_COLOR,
            width=4.0,
            opacity=1.0,
        )
        front.set_z_index(24.1)
        group.add(front)
    return group


def build_switch_diagram(progress: object) -> VGroup:
    """Build one deterministic visual frame for playback or test capture."""

    frame = compute_switch_frame(progress)
    axis = DashedLine(
        project_point((0.0, 0.0, AXIAL_RANGE[0] - 0.15)),
        project_point((0.0, 0.0, AXIAL_RANGE[1] + 0.15)),
        dash_length=0.10,
        color="#A8BED0",
        stroke_width=1.15,
        stroke_opacity=0.30,
    )
    axis.set_z_index(4.0)
    return VGroup(
        axis,
        _plane_group(),
        _surface_group(frame),
        _sphere_group(frame),
        _section_group(frame),
    )


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
    "PLANE_NORMAL",
    "PLANE_OFFSET",
    "SURFACE_RADIUS",
    "build_switch_diagram",
    "compute_switch_frame",
    "project_point",
    "section_point",
)


config.background_color = BACKGROUND_COLOR
