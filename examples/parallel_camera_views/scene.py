from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    GOLD,
    GREEN,
    RED,
    WHITE,
    Dot3D,
    Line,
    Polygon,
    ThreeDScene,
    VGroup,
)

from tikz_native.camera_3d import MultiProjectionCamera
from tikz_native.parallel_camera import CameraPlane, ParallelCameraState


class _ParallelCameraScene(ThreeDScene):
    def __init__(self, **kwargs) -> None:
        super().__init__(camera_class=MultiProjectionCamera, **kwargs)

    def move_camera_to(
        self,
        state: ParallelCameraState,
        *,
        run_time: float = 1.0,
        arc_height: float = 0.55,
        transition: str = "orbit",
    ) -> None:
        self.play(
            self.camera.animate_to_parallel_state(
                state,
                transition=transition,
                arc_height=arc_height,
            ),
            run_time=run_time,
        )


def _wire_box(
    center: np.ndarray,
    half_size: tuple[float, float, float],
) -> VGroup:
    offsets = [
        np.array((x, y, z), dtype=float)
        for x in (-half_size[0], half_size[0])
        for y in (-half_size[1], half_size[1])
        for z in (-half_size[2], half_size[2])
    ]
    vertices = [center + offset for offset in offsets]
    edges = []
    for first in range(len(offsets)):
        for second in range(first + 1, len(offsets)):
            changed_axes = np.count_nonzero(
                np.abs(offsets[first] - offsets[second]) > 1.0e-12
            )
            if changed_axes == 1:
                edges.append(
                    Line(
                        vertices[first],
                        vertices[second],
                        color=BLUE,
                        stroke_width=4.0,
                    )
                )
    return VGroup(*edges)


def _plane_geometry(plane: CameraPlane) -> VGroup:
    center = plane.point
    corners = (
        center - 2.4 * plane.u_axis - 1.35 * plane.v_axis,
        center + 2.4 * plane.u_axis - 1.35 * plane.v_axis,
        center + 2.4 * plane.u_axis + 1.35 * plane.v_axis,
        center - 2.4 * plane.u_axis + 1.35 * plane.v_axis,
    )
    patch = Polygon(
        *corners,
        color=BLUE,
        fill_color=BLUE,
        fill_opacity=0.22,
        stroke_width=4.0,
    )
    axes = (
        Line(center, center + 1.7 * plane.u_axis, color=RED, stroke_width=6.0),
        Line(center, center + 1.7 * plane.v_axis, color=GREEN, stroke_width=6.0),
        Line(center, center + 1.7 * plane.normal, color=GOLD, stroke_width=6.0),
    )
    corner_markers = tuple(
        Dot3D(corner, radius=0.065, color=color)
        for corner, color in zip(corners, (RED, GREEN, GOLD, WHITE), strict=True)
    )
    above = center + 0.9 * plane.normal
    below = center - 0.9 * plane.normal
    normal_reference = (
        Line(below, above, color=WHITE, stroke_width=3.0),
        Dot3D(above, radius=0.08, color=GOLD),
        Dot3D(below, radius=0.08, color=RED),
    )
    return VGroup(
        patch,
        *axes,
        *corner_markers,
        *normal_reference,
        Dot3D(center, radius=0.08, color=WHITE),
    )


def _screen_crosshair(position: np.ndarray, *, color=WHITE) -> VGroup:
    center = np.array((position[0], position[1], 0.0), dtype=float)
    return VGroup(
        Line(
            center + np.array((-0.18, 0.0, 0.0)),
            center + np.array((0.18, 0.0, 0.0)),
            color=color,
            stroke_width=3.0,
        ),
        Line(
            center + np.array((0.0, -0.18, 0.0)),
            center + np.array((0.0, 0.18, 0.0)),
            color=color,
            stroke_width=3.0,
        ),
    )


class TargetOrbitCameraDemo(_ParallelCameraScene):
    """Move between two authored world targets while the view also rotates."""

    def construct(self) -> None:
        center = np.array((0.55, -0.35, 0.4))
        first_target = center + np.array((-1.25, -0.55, 0.65))
        second_target = center + np.array((1.35, 0.6, -0.45))
        self.add(
            _wire_box(center, (2.45, 1.5, 1.15)),
            Dot3D(first_target, radius=0.1, color=RED),
            Dot3D(second_target, radius=0.1, color=GOLD),
        )

        first_view = ParallelCameraState.from_view_direction(
            (1.25, -1.0, 0.8),
            up_hint=(0.0, 0.0, 1.0),
            target=first_target,
            zoom=0.92,
        )
        second_view = ParallelCameraState.from_view_direction(
            (-1.1, -0.65, 1.0),
            up_hint=(0.0, 0.0, 1.0),
            target=second_target,
            zoom=1.08,
        )
        final_view = ParallelCameraState.from_view_direction(
            (0.65, 1.2, 0.75),
            up_hint=(0.0, 0.0, 1.0),
            target=center,
            zoom=0.86,
        )

        self.camera.set_parallel_state(first_view)
        self.wait(0.4)
        self.move_camera_to(second_view, run_time=1.25)
        self.wait(0.35)
        self.move_camera_to(final_view, run_time=1.1)
        self.wait(0.4)


class PlaneViewReductionDemo(_ParallelCameraScene):
    """Reduce one finite plane to a certified line, then restore its area."""

    def construct(self) -> None:
        plane = CameraPlane(
            point=np.array((0.8, -0.45, 0.55)),
            normal=np.array((0.55, -0.35, 1.0)),
            u_axis=np.array((1.0, 1.0, 0.0)),
        )
        self.add(_plane_geometry(plane))

        normal = ParallelCameraState.normal_to_plane(plane, zoom=1.05)
        relative = ParallelCameraState.relative_to_plane(
            plane,
            inclination_degrees=58.0,
            azimuth_degrees=32.0,
            zoom=1.05,
        )
        near_edge = ParallelCameraState.relative_to_plane(
            plane,
            inclination_degrees=89.0,
            azimuth_degrees=32.0,
            zoom=1.05,
        )
        edge_on = ParallelCameraState.along_plane(
            plane,
            azimuth_degrees=32.0,
            zoom=1.05,
        )

        self.camera.set_parallel_state(normal)
        self.wait(0.4)
        self.move_camera_to(relative, run_time=0.9, transition="shortest")
        self.wait(0.3)
        self.move_camera_to(near_edge, run_time=0.75, transition="shortest")
        self.wait(0.3)
        self.move_camera_to(edge_on, run_time=0.7, transition="shortest")
        self.wait(0.65)
        self.move_camera_to(near_edge, run_time=0.7, transition="shortest")
        self.move_camera_to(normal, run_time=1.0, transition="shortest")
        self.wait(0.4)


class AnchorZoomCameraDemo(_ParallelCameraScene):
    """Isolate target, viewport-anchor, and the two independent zoom layers."""

    def construct(self) -> None:
        center = np.array((0.65, -0.4, 0.35))
        first_target = center + np.array((-1.15, -0.45, 0.55))
        second_target = center + np.array((1.25, 0.55, -0.35))
        left_anchor = np.array((-2.35, 0.85))
        right_anchor = np.array((2.1, -0.65))
        frame_center = np.array((0.8, -0.35, 0.0))
        self.camera.frame_center = frame_center
        self.add(
            _wire_box(center, (2.2, 1.35, 1.0)),
            Dot3D(first_target, radius=0.11, color=RED),
            Dot3D(second_target, radius=0.11, color=GOLD),
            Line(first_target, second_target, color=WHITE, stroke_width=3.0),
        )
        crosshairs = (
            _screen_crosshair(frame_center[:2] + left_anchor, color=RED),
            _screen_crosshair(frame_center[:2] + right_anchor, color=GOLD),
        )
        self.add_fixed_in_frame_mobjects(*crosshairs)

        first = ParallelCameraState.from_view_direction(
            (1.0, -1.2, 0.8),
            up_hint=(0.0, 0.0, 1.0),
            target=first_target,
            screen_anchor=left_anchor,
            zoom=0.65,
        )
        second = ParallelCameraState(
            first.matrix,
            target=second_target,
            screen_anchor=left_anchor,
            zoom=0.65,
        )
        anchored = ParallelCameraState(
            first.matrix,
            target=second_target,
            screen_anchor=right_anchor,
            zoom=1.35,
        )
        rotated = ParallelCameraState.from_view_direction(
            (-0.75, -1.15, 1.0),
            up_hint=(0.0, 0.0, 1.0),
            target=second_target,
            screen_anchor=right_anchor,
            zoom=1.35,
        )

        self.camera.set_parallel_state(first)
        self.wait(0.4)
        self.move_camera_to(second, run_time=1.0, transition="shortest")
        self.wait(0.3)
        self.move_camera_to(anchored, run_time=1.1, transition="shortest")
        self.wait(0.35)
        self.play(self.camera.zoom_tracker.animate.set_value(1.35), run_time=0.9)
        self.wait(0.35)
        self.move_camera_to(rotated, run_time=1.0)
        self.wait(0.45)


class FrameCenterCompatibilityDemo(_ParallelCameraScene):
    """Exercise semantic-to-legacy orbit and snapshot restore after scaling."""

    def construct(self) -> None:
        center = np.array((0.55, -0.3, 0.35))
        frame_center = np.array((1.05, -0.65, 0.2))
        anchor = np.array((-1.8, 0.7))
        self.camera.frame_center = frame_center
        self.camera.set_zoom(1.25)
        self.add(
            _wire_box(center, (2.15, 1.25, 0.95)),
            Dot3D(center, radius=0.11, color=WHITE),
        )
        marker = _screen_crosshair(frame_center[:2] + anchor, color=WHITE)
        self.add_fixed_in_frame_mobjects(marker)

        semantic = ParallelCameraState.from_view_direction(
            (0.9, -1.25, 0.85),
            up_hint=(0.0, 0.0, 1.0),
            target=center,
            screen_anchor=anchor,
            zoom=1.1,
        )
        self.camera.set_parallel_state(semantic)
        snapshot = self.camera.snapshot()
        self.wait(0.45)
        self.play(self.camera.animate_orbit_to("front"), run_time=1.0)
        self.wait(0.35)
        self.camera.restore(snapshot)
        self.wait(0.55)


class ParallelCameraViewsDemo(_ParallelCameraScene):
    """One compact sequence covering every semantic plane-view constructor."""

    def construct(self) -> None:
        plane = CameraPlane(
            point=np.array((0.8, -0.45, 0.55)),
            normal=np.array((0.55, -0.35, 1.0)),
            u_axis=np.array((1.0, 1.0, 0.0)),
        )
        center = plane.point
        self.add(_plane_geometry(plane))

        normal = ParallelCameraState.normal_to_plane(plane, target=center)
        relative = ParallelCameraState.relative_to_plane(
            plane,
            inclination_degrees=58.0,
            azimuth_degrees=32.0,
            target=center,
        )
        edge_on = ParallelCameraState.along_plane(
            plane,
            azimuth_degrees=32.0,
            target=center,
        )
        anchored = ParallelCameraState.normal_to_plane(
            plane,
            target=center,
            screen_anchor=(-2.25, 0.6),
            zoom=1.15,
        )

        self.wait(0.35)
        for index, state in enumerate((normal, relative, edge_on, anchored)):
            if index == 3:
                # The final anchor is viewport-relative and therefore remains
                # authoritative even when Manim's inherited frame moves.
                self.camera.frame_center = np.array((0.8, -0.35, 0.0))
            self.move_camera_to(state)
            self.wait(0.25)
