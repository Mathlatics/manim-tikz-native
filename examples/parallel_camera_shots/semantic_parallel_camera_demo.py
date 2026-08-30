"""Short Cairo demos for semantic parallel-camera shot authoring.

Each scene is intentionally independent so it can be rendered as a small
acceptance clip.  The two cone scenes feed the *same* live camera state into
the automatic-occlusion controllers; there is no separately-authored display
projection.

Quick preview::

    PYTHONPATH=. manim --renderer=cairo -ql -r 480,270 --fps 6 \
        examples/parallel_camera_shots/semantic_parallel_camera_demo.py \
        SemanticPlaneShotDemo
"""

from __future__ import annotations

from math import pi
from typing import Mapping

import numpy as np
from manim import (
    BLUE,
    GOLD,
    GREEN,
    RED,
    UP,
    WHITE,
    Dot3D,
    Line,
    Polygon,
    Text,
    ThreeDScene,
    VGroup,
)

from polyhedron_visibility.quadrics.authoring import QuadricSection3D
from polyhedron_visibility.quadrics.composite_authoring import (
    CompositeQuadricSection3D,
)
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricManimLimits,
    QuadricManimStyle,
)
from tikz_native.camera_3d import MultiProjectionCamera
from tikz_native.parallel_camera import CameraPlane
from tikz_native.parallel_shots import (
    ParallelCameraShot,
    ParallelCameraShotSequence,
)
from tikz_native.parallel_shots_manim import play_parallel_camera_shot


BACKGROUND = "#091521"
LABEL_COLOR = "#EAF2F8"
MUTED_COLOR = "#9FB3C5"
SECTION_COLOR = "#FFD166"


class _SemanticParallelShotScene(ThreeDScene):
    """Shared camera, status overlay, and sequence playback helpers."""

    def __init__(self, **kwargs) -> None:
        super().__init__(camera_class=MultiProjectionCamera, **kwargs)

    def _install_overlay(
        self,
        first_shot: ParallelCameraShot,
        *,
        topology: str,
    ) -> tuple[Text, VGroup, Dot3D]:
        label = _shot_label(first_shot, topology=topology)
        crosshair = _screen_crosshair(first_shot.state.screen_anchor)
        label.set_z_index(1000)
        crosshair.set_z_index(1000)
        self.add_fixed_in_frame_mobjects(label, crosshair)

        target_marker = Dot3D(
            first_shot.state.target,
            radius=0.075,
            color=GOLD,
        )

        def follow_target(marker: Dot3D) -> None:
            marker.move_to(self.camera.snapshot_parallel_state().target)

        def follow_anchor(marker: VGroup) -> None:
            state = self.camera.snapshot_parallel_state()
            marker.become(_screen_crosshair(state.screen_anchor))
            marker.set_z_index(1000)

        target_marker.add_updater(follow_target)
        crosshair.add_updater(follow_anchor)
        self.add(target_marker)
        return label, crosshair, target_marker

    def _play_sequence(
        self,
        sequence: ParallelCameraShotSequence,
        *,
        topology_by_id: Mapping[str, str],
    ) -> None:
        """Play one validated sequence while keeping its cue label in sync."""

        first = sequence.shots[0]
        self.camera.set_parallel_state(first.state)
        label, crosshair, target_marker = self._install_overlay(
            first,
            topology=f"STATE {topology_by_id[first.id]}",
        )
        if first.hold > 0.0:
            self.wait(first.hold)

        for shot in sequence.shots[1:]:
            label.become(
                _shot_label(
                    shot,
                    topology=f"TARGET {topology_by_id[shot.id]}",
                )
            )
            label.set_z_index(1000)
            play_parallel_camera_shot(self, shot)

        crosshair.clear_updaters()
        target_marker.clear_updaters()


def _shot_label(shot: ParallelCameraShot, *, topology: str) -> Text:
    state = shot.state
    target = ", ".join(f"{value:.2f}" for value in state.target)
    anchor = ", ".join(f"{value:.2f}" for value in state.screen_anchor)
    cue = shot.cue or shot.id
    return Text(
        f"{topology}  |  {cue}\n"
        f"target=({target})   anchor=({anchor})   zoom={state.zoom:.2f}",
        font_size=22,
        color=LABEL_COLOR,
        line_spacing=0.82,
    ).to_edge(UP, buff=0.16)


def _screen_crosshair(position: np.ndarray) -> VGroup:
    center = np.array((position[0], position[1], 0.0), dtype=float)
    extent = 0.13
    return VGroup(
        Line(
            center + (-extent, 0.0, 0.0),
            center + (extent, 0.0, 0.0),
            color=GOLD,
            stroke_width=2.6,
        ),
        Line(
            center + (0.0, -extent, 0.0),
            center + (0.0, extent, 0.0),
            color=GOLD,
            stroke_width=2.6,
        ),
    )


def _finite_plane_geometry(plane: CameraPlane) -> VGroup:
    center = plane.point
    corners = (
        center - 2.25 * plane.u_axis - 1.22 * plane.v_axis,
        center + 2.25 * plane.u_axis - 1.22 * plane.v_axis,
        center + 2.25 * plane.u_axis + 1.22 * plane.v_axis,
        center - 2.25 * plane.u_axis + 1.22 * plane.v_axis,
    )
    patch = Polygon(
        *corners,
        color=BLUE,
        fill_color=BLUE,
        fill_opacity=0.24,
        stroke_width=3.2,
    )
    axes = VGroup(
        Line(center, center + 1.65 * plane.u_axis, color=RED, stroke_width=5.0),
        Line(center, center + 1.65 * plane.v_axis, color=GREEN, stroke_width=5.0),
        Line(center, center + 1.35 * plane.normal, color=GOLD, stroke_width=5.0),
    )
    frame = VGroup(
        *(Line(corners[index], corners[(index + 1) % 4], color=WHITE)
          for index in range(4))
    )
    return VGroup(patch, frame, axes)


def _preview_limits(*, surface_count: int) -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=surface_count,
        max_curves=8,
        max_fragments_per_curve=24,
        max_segments_per_fragment=192,
        max_surface_segments=320,
        max_dashes_per_fragment=64,
        max_projected_length=24.0,
        max_total_mobjects=30000,
        max_boundary_sources=32,
    )


def _quadric_style() -> QuadricManimStyle:
    return QuadricManimStyle(
        surface_fill_color="#2B6F9F",
        surface_fill_opacity=0.68,
        surface_stroke_color="#4FA3CE",
        surface_stroke_opacity=0.28,
        visible_curve_color=SECTION_COLOR,
        visible_curve_width=4.0,
        hidden_curve_color=SECTION_COLOR,
        hidden_curve_width=2.7,
        hidden_curve_opacity=0.48,
        section_plane_fill_color="#28B7A0",
        section_plane_fill_opacity=0.17,
        section_plane_stroke_color="#75E0D0",
        section_plane_stroke_opacity=0.78,
        cone_lateral_fill_colors=("#173753", "#4F9AC1", "#1D4368"),
    )


class SemanticPlaneShotDemo(_SemanticParallelShotScene):
    """normal -> relative -> along -> return for an ordinary finite plane."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        plane = CameraPlane(
            point=np.array((0.45, -0.28, 0.34)),
            normal=np.array((0.52, -0.34, 1.0)),
            u_axis=np.array((1.0, 0.72, -0.27)),
        )
        self.add(_finite_plane_geometry(plane))

        sequence = ParallelCameraShotSequence(
            (
                ParallelCameraShot.normal_to_plane(
                    "normal",
                    plane,
                    target=plane.point,
                    screen_anchor=(-0.55, -0.15),
                    zoom=1.00,
                    duration=0.7,
                    hold=0.35,
                    transition="shortest",
                    cue="normal to plane",
                ),
                ParallelCameraShot.relative_to_plane(
                    "relative",
                    plane,
                    inclination_degrees=48.0,
                    azimuth_degrees=30.0,
                    target=plane.point + 0.36 * plane.u_axis,
                    screen_anchor=(0.42, -0.28),
                    zoom=0.84,
                    duration=0.9,
                    hold=0.25,
                    cue="relative 48 deg",
                ),
                ParallelCameraShot.along_plane(
                    "along",
                    plane,
                    azimuth_degrees=30.0,
                    target=plane.point - 0.24 * plane.v_axis,
                    screen_anchor=(-0.28, 0.12),
                    zoom=1.12,
                    duration=0.85,
                    hold=0.45,
                    transition="shortest",
                    cue="along plane: exact LINE",
                ),
                ParallelCameraShot.normal_to_plane(
                    "return",
                    plane,
                    target=plane.point,
                    screen_anchor=(0.0, 0.0),
                    zoom=0.96,
                    duration=0.9,
                    hold=0.35,
                    cue="return",
                ),
            )
        )
        self._play_sequence(
            sequence,
            topology_by_id={
                "normal": "AREA",
                "relative": "AREA",
                "along": "LINE",
                "return": "AREA",
            },
        )


class SingleConeSectionShotDemo(_SemanticParallelShotScene):
    """One closed finite cone section through AREA -> LINE -> AREA."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        cone = ConeSpec(
            "shot-demo:single-cone",
            (0.0, 0.0, -1.5),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 4.0),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.CLOSED_SINGLE,
        )
        plane = SectionPlane(
            "shot-demo:single-plane",
            (0.0, 0.0, 0.0),
            (0.35, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        sequence = ParallelCameraShotSequence(
            (
                ParallelCameraShot.relative_to_plane(
                    "single-area-in",
                    plane,
                    inclination_degrees=18.0,
                    azimuth_degrees=37.0,
                    target=plane.point,
                    screen_anchor=(-0.18, 0.12),
                    zoom=0.92,
                    duration=0.7,
                    hold=0.35,
                    cue="single cone: AREA",
                ),
                ParallelCameraShot.along_plane(
                    "single-line",
                    plane,
                    direction=(0.0, -1.0, 0.0),
                    target=plane.point,
                    screen_anchor=(0.16, -0.11),
                    zoom=1.06,
                    duration=0.9,
                    hold=0.45,
                    cue="edge-on: LINE",
                ),
                ParallelCameraShot.relative_to_plane(
                    "single-area-out",
                    plane,
                    inclination_degrees=31.0,
                    azimuth_degrees=23.0,
                    target=(0.08, -0.12, 0.18),
                    screen_anchor=(-0.14, 0.17),
                    zoom=0.88,
                    duration=0.9,
                    hold=0.4,
                    cue="single cone: AREA",
                ),
            )
        )
        self.camera.set_parallel_state(sequence.shots[0].state)
        QuadricSection3D(
            self,
            surface=cone,
            section_id="shot-demo:single-section",
            plane=plane,
            projection=lambda active_scene: active_scene.camera,
            draw_section_boundary=True,
            show_plane=True,
            include_surface_boundaries=True,
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            style=_quadric_style(),
            limits=_preview_limits(surface_count=1),
            max_chord_error=0.035,
            section_max_screen_error=0.16,
            plane_patch_margin=0.16,
        ).attach()
        self._play_sequence(
            sequence,
            topology_by_id={
                "single-area-in": "AREA",
                "single-line": "LINE",
                "single-area-out": "AREA",
            },
        )


class OpenDoubleSectionShotDemo(_SemanticParallelShotScene):
    """Open-double offset hyperbola with live automatic occlusion."""

    def construct(self) -> None:
        self.camera.background_color = BACKGROUND
        cone = ConeSpec(
            "shot-demo:open-double",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (-2.0, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.OPEN_DOUBLE,
        )
        plane = SectionPlane(
            "shot-demo:double-plane",
            (0.0, 0.48, 0.0),
            (0.0, 1.0, 0.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        sequence = ParallelCameraShotSequence(
            (
                ParallelCameraShot.relative_to_plane(
                    "double-area-in",
                    plane,
                    inclination_degrees=14.0,
                    azimuth_degrees=0.0,
                    target=plane.point,
                    screen_anchor=(-0.12, 0.09),
                    zoom=0.93,
                    duration=0.7,
                    hold=0.35,
                    cue="OPEN_DOUBLE: AREA",
                ),
                ParallelCameraShot.along_plane(
                    "double-line",
                    plane,
                    direction=(1.0, 0.0, 0.0),
                    target=plane.point,
                    screen_anchor=(0.14, -0.09),
                    zoom=1.04,
                    duration=0.9,
                    hold=0.5,
                    cue="OPEN_DOUBLE: LINE",
                ),
                ParallelCameraShot.relative_to_plane(
                    "double-area-out",
                    plane,
                    inclination_degrees=23.0,
                    azimuth_degrees=0.0,
                    target=(0.11, 0.40, 0.17),
                    screen_anchor=(-0.15, 0.13),
                    zoom=0.89,
                    duration=0.9,
                    hold=0.4,
                    cue="OPEN_DOUBLE: AREA",
                ),
            )
        )
        self.camera.set_parallel_state(sequence.shots[0].state)
        CompositeQuadricSection3D(
            self,
            surface=cone,
            section_id="shot-demo:double-section",
            plane=plane,
            projection=lambda active_scene: active_scene.camera,
            draw_section_boundary=True,
            include_surface_boundaries=True,
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            style=_quadric_style(),
            limits=_preview_limits(surface_count=2),
            max_chord_error=0.04,
            section_max_screen_error=0.16,
            plane_patch_margin=0.16,
        ).attach()
        self._play_sequence(
            sequence,
            topology_by_id={
                "double-area-in": "AREA + OCCLUSION",
                "double-line": "LINE + OCCLUSION",
                "double-area-out": "AREA + OCCLUSION",
            },
        )


__all__ = [
    "OpenDoubleSectionShotDemo",
    "SemanticPlaneShotDemo",
    "SingleConeSectionShotDemo",
]
