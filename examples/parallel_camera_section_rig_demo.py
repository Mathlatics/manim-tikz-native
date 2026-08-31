"""Small Cairo acceptance scenes for the parallel-camera section Rig.

These scenes use the complete certified path: semantic display catalog,
shot/timeline compilation, one-controller Rig binding, coordinated playback,
and deterministic cleanup.  They intentionally contain no text or LaTeX.

Quick previews::

    PYTHONPATH=. manim --renderer=cairo -ql --fps 6 \
        examples/parallel_camera_section_rig_demo.py \
        ParallelCameraSectionMotionDemo

    PYTHONPATH=. manim --renderer=cairo -ql --fps 6 \
        examples/parallel_camera_section_rig_demo.py \
        ParallelCameraPlaneRankReductionDemo

    PYTHONPATH=. manim --renderer=cairo -ql --fps 12 \
        examples/parallel_camera_section_rig_demo.py \
        ParallelCameraConeTopologyDemo
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import pi

from manim import ThreeDScene, config

from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricManimLimits,
    QuadricManimStyle,
)
from polyhedron_visibility.quadrics.parallel_plane_motion import (
    ParallelPlaneTranslation,
)
from polyhedron_visibility.quadrics.plane_motion import AxisAnglePlaneMotion
from polyhedron_visibility.quadrics.section_timeline import (
    SectionTimeline,
    compile_section_timeline,
)
from polyhedron_visibility.quadrics.semantic_display import (
    SectionDisplayInstruction,
    compile_section_display,
)
from tikz_native.camera_3d import MultiProjectionCamera, OBLIQUE_MATRIX
from tikz_native.parallel_camera import ParallelCameraState
from tikz_native.parallel_preflight import (
    ParallelPreflightLimits,
    ParallelSafeFrame,
)
from tikz_native.parallel_shots import (
    ParallelCameraShot,
    ParallelCameraShotSequence,
)
from tikz_native.quadric_section_parallel_manim import (
    play_parallel_section_sequence,
)
from tikz_native.quadric_section_parallel_rig import (
    ParallelSectionRigBinding,
    build_parallel_section_rig_display_catalog,
    compile_parallel_section_rig_from_shots,
)


def _preflight_limits() -> ParallelPreflightLimits:
    return ParallelPreflightLimits(
        ParallelSafeFrame(-6.5, 6.5, -3.5, 3.5),
        min_zoom=0.45,
        max_zoom=1.4,
    )


def _controller_options() -> dict[str, object]:
    """Keep low-quality previews light while retaining the real painter graph."""

    return {
        "limits": QuadricManimLimits(
            max_surfaces=2,
            max_curves=12,
            max_fragments_per_curve=12,
            max_segments_per_fragment=384,
            max_surface_segments=384,
            max_dashes_per_fragment=128,
            max_projected_length=20.0,
            max_total_mobjects=8000,
            max_boundary_sources=16,
            max_boundary_styles=16,
        ),
        "style": QuadricManimStyle(
            surface_fill_color="#2457A7",
            surface_fill_opacity=0.34,
            surface_stroke_color="#4F8EE8",
            surface_stroke_width=1.8,
            visible_curve_color="#FFE066",
            visible_curve_width=4.0,
            hidden_curve_color="#F5C84C",
            hidden_curve_width=2.5,
            hidden_curve_opacity=0.72,
            section_plane_fill_color="#58C7B4",
            section_plane_fill_opacity=0.20,
            section_plane_stroke_color="#9AF0DE",
            section_plane_stroke_width=2.0,
            section_plane_stroke_opacity=0.90,
        ),
    }


def _cone_controller_options() -> dict[str, object]:
    """Reserve the larger finite-branch capacity needed by the cone sweep."""

    options = _controller_options()
    options["limits"] = QuadricManimLimits(
        max_surfaces=2,
        max_curves=16,
        max_fragments_per_curve=16,
        max_segments_per_fragment=512,
        max_surface_segments=512,
        max_dashes_per_fragment=128,
        max_projected_length=30.0,
        max_total_mobjects=20000,
        max_boundary_sources=16,
        max_boundary_styles=16,
    )
    return options


class _ParallelSectionRigScene(ThreeDScene):
    def __init__(self, **kwargs) -> None:
        super().__init__(camera_class=MultiProjectionCamera, **kwargs)

    def _play_certified_rig(
        self,
        timeline: SectionTimeline,
        shots: ParallelCameraShotSequence,
        initial_camera: ParallelCameraState,
        *,
        semantic_bank_ids: tuple[str, str],
        controller_options: dict[str, object] | None = None,
        acceptance: Callable[[ParallelSectionRigBinding], None] | None = None,
        transition_fraction: float = 0.25,
    ) -> None:
        catalog = build_parallel_section_rig_display_catalog(
            timeline,
            semantic_bank_ids,
            include_plane=True,
        )
        display = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("painted"),
        )
        display_frames: Sequence = tuple(display for _ in timeline.samples)

        self.camera.set_parallel_state(initial_camera)
        binding = None
        coordinator = None
        try:
            binding = compile_parallel_section_rig_from_shots(
                self,
                timeline,
                shots,
                initial_camera,
                display_frames,
                limits=_preflight_limits(),
                semantic_bank_ids=semantic_bank_ids,
                frame_rate=float(config.frame_rate),
                plane_patch_margin=0.16,
                transition_fraction=transition_fraction,
                controller_options=(
                    _controller_options()
                    if controller_options is None
                    else controller_options
                ),
            )
            binding.attach()
            coordinator = binding.build_coordinator(self.camera)
            play_parallel_section_sequence(
                self,
                binding.sequence,
                shots,
                coordinator,
            )
            if acceptance is not None:
                acceptance(binding)
            # Retain the exact authored endpoint long enough to inspect it.
            self.wait(0.5)
        finally:
            try:
                if coordinator is not None and coordinator.active:
                    coordinator.restore()
            finally:
                if binding is not None:
                    binding.restore()


class ParallelCameraSectionMotionDemo(_ParallelSectionRigScene):
    """Move target, screen anchor, zoom, projection, and the section plane."""

    def construct(self) -> None:
        sphere = SphereSpec("motion-sphere", (0.0, 0.0, 0.0), 1.35)
        plane = SectionPlane(
            "motion-plane",
            (0.0, 0.0, -0.55),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        timeline = compile_section_timeline(
            "motion-section",
            sphere,
            (
                ParallelPlaneTranslation(
                    "motion-plane-translation",
                    plane,
                    (0.0, 0.0, 1.10),
                    start_time=0.0,
                    end_time=2.0,
                ),
            ),
        )

        initial = ParallelCameraState(
            OBLIQUE_MATRIX,
            target=(-0.35, -0.18, -0.30),
            screen_anchor=(-0.65, 0.28),
            zoom=0.72,
        )
        middle = ParallelCameraState.from_view_direction(
            (-0.75, 1.0, 0.95),
            target=(0.08, 0.18, 0.0),
            screen_anchor=(0.55, -0.22),
            zoom=1.02,
        )
        final = ParallelCameraState(
            OBLIQUE_MATRIX,
            target=(0.38, 0.08, 0.42),
            screen_anchor=(-0.30, 0.15),
            zoom=0.84,
        )
        shots = ParallelCameraShotSequence(
            (
                ParallelCameraShot(
                    "motion-to-free-view",
                    middle,
                    duration=1.0,
                    transition="orbit",
                    arc_height=0.50,
                ),
                ParallelCameraShot(
                    "motion-back-to-oblique",
                    final,
                    duration=1.0,
                    transition="orbit",
                    arc_height=-0.45,
                ),
            )
        )
        self._play_certified_rig(
            timeline,
            shots,
            initial,
            semantic_bank_ids=("motion-bank-a", "motion-bank-b"),
        )


class ParallelCameraPlaneRankReductionDemo(_ParallelSectionRigScene):
    """Move from oblique to plane-normal and finally exact edge-on view."""

    def construct(self) -> None:
        sphere = SphereSpec("rank-sphere", (0.0, 0.0, 0.0), 1.35)
        plane = SectionPlane(
            "rank-plane",
            (0.0, 0.0, -0.30),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        motion = ParallelPlaneTranslation(
            "rank-plane-translation",
            plane,
            (0.0, 0.0, 0.40),
            start_time=0.0,
            end_time=1.6,
        )
        timeline = compile_section_timeline(
            "rank-section",
            sphere,
            (motion,),
        )

        initial_plane = motion.plane_at(0.0)
        middle_plane = motion.plane_at(0.5)
        final_plane = motion.plane_at(1.0)
        initial = ParallelCameraState.relative_to_plane(
            initial_plane,
            inclination_degrees=48.0,
            azimuth_degrees=30.0,
            target=initial_plane.point,
            zoom=0.88,
        )
        normal = ParallelCameraState.normal_to_plane(
            middle_plane,
            target=middle_plane.point,
            zoom=0.94,
        )
        edge_on = ParallelCameraState.along_plane(
            final_plane,
            direction=(1.0, 0.0, 0.0),
            target=final_plane.point,
            zoom=0.94,
        )
        shots = ParallelCameraShotSequence(
            (
                ParallelCameraShot(
                    "rank-normal-to-plane",
                    normal,
                    duration=0.8,
                    transition="shortest",
                ),
                ParallelCameraShot(
                    "rank-exact-edge-on",
                    edge_on,
                    duration=0.8,
                    transition="shortest",
                ),
            )
        )
        self._play_certified_rig(
            timeline,
            shots,
            initial,
            semantic_bank_ids=("rank-bank-a", "rank-bank-b"),
        )


class ParallelCameraConeTopologyDemo(_ParallelSectionRigScene):
    """Cross ellipse/parabola/hyperbola banks and finish exactly edge-on."""

    def construct(self) -> None:
        cone = ConeSpec(
            "topology-cone",
            (0.0, 0.0, -1.5),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 4.0),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.CLOSED_SINGLE,
        )
        plane = SectionPlane(
            "topology-plane",
            (0.0, 0.0, 0.2),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        motion = AxisAnglePlaneMotion(
            "topology-plane-rotation",
            plane,
            (0.0, 0.0, 0.2),
            (0.0, 1.0, 0.0),
            0.0,
            1.2,
            start_time=0.0,
            end_time=6.0,
        )
        timeline = compile_section_timeline(
            "topology-section",
            cone,
            (motion,),
        )

        initial = ParallelCameraState.from_view_direction(
            (1.0, 1.0, 1.0),
            target=(0.0, 0.0, 0.45),
            screen_anchor=(-0.18, 0.02),
            zoom=0.82,
        )
        final_plane = motion.plane_at(1.0)
        edge_on = ParallelCameraState.along_plane(
            final_plane,
            direction=final_plane.u_axis,
            target=final_plane.point,
            screen_anchor=(0.10, 0.0),
            zoom=0.82,
        )
        shots = ParallelCameraShotSequence(
            (
                # Keep the proven oblique view through both analytic topology
                # events; the short final shot then demonstrates a real orbit
                # into exact edge-on projection without passing a trim
                # tangency while the camera itself is also moving.
                ParallelCameraShot(
                    "topology-hold-oblique",
                    initial,
                    duration=5.4,
                    transition="shortest",
                ),
                ParallelCameraShot(
                    "topology-orbit-to-edge-on",
                    edge_on,
                    duration=0.6,
                    transition="orbit",
                    arc_height=0.50,
                ),
            )
        )

        def require_complete_topology_path(
            binding: ParallelSectionRigBinding,
        ) -> None:
            families = {
                frame.signature.conic_family.value
                for frame in timeline.animation.frames
            }
            if not {"oval", "parabola", "hyperbola"}.issubset(families):
                raise RuntimeError(
                    "cone demo did not certify oval/parabola/hyperbola families"
                )
            crossfades = tuple(
                frame
                for frame in binding.sequence.bank_render_frames
                if len(frame.layers) == 2
            )
            if not crossfades:
                raise RuntimeError("cone demo did not activate a two-bank crossfade")
            used_banks = {
                layer.semantic_bank_id
                for frame in binding.sequence.bank_render_frames
                for layer in frame.layers
            }
            if used_banks != {"topology-bank-a", "topology-bank-b"}:
                raise RuntimeError("cone demo did not exercise both topology banks")
            if not any(
                layer.active_cap_chord_ids
                for frame in binding.sequence.bank_render_frames
                for layer in frame.layers
            ):
                raise RuntimeError("cone demo did not activate a finite cap chord")
            final_section = binding.controller.last_section_frame
            if (
                final_section is None
                or final_section.projection_kind.value != "line"
            ):
                raise RuntimeError("cone demo did not commit the final rank-one line")

        self._play_certified_rig(
            timeline,
            shots,
            initial,
            semantic_bank_ids=("topology-bank-a", "topology-bank-b"),
            controller_options=_cone_controller_options(),
            acceptance=require_complete_topology_path,
            transition_fraction=0.5,
        )


__all__ = [
    "ParallelCameraConeTopologyDemo",
    "ParallelCameraPlaneRankReductionDemo",
    "ParallelCameraSectionMotionDemo",
]
