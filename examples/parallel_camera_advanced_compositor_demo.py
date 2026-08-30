"""Cairo acceptance scenes for the advanced parallel-camera compositor.

Quick previews::

    PYTHONPATH=. manim --renderer=cairo -ql --fps 12 \
        examples/parallel_camera_advanced_compositor_demo.py \
        ParallelViewportTangentPointDemo

    PYTHONPATH=. manim --renderer=cairo -ql --fps 12 \
        examples/parallel_camera_advanced_compositor_demo.py \
        ParallelCompositingAxesDemo

    PYTHONPATH=. manim --renderer=cairo -ql --fps 12 \
        examples/parallel_camera_advanced_compositor_demo.py \
        GlobalParallelRigOcclusionDemo

The scenes contain no text or LaTeX.  Each one exercises the production
transaction path and raises if its final semantic evidence is missing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from manim import ThreeDScene, ValueTracker, VMobject, config, linear

from polyhedron_visibility.quadrics.contract import SectionPlane, SphereSpec
from polyhedron_visibility.quadrics.manim import (
    QuadricManimLimits,
    QuadricManimStyle,
)
from polyhedron_visibility.quadrics.parallel_plane_motion import (
    ParallelPlaneTranslation,
)
from polyhedron_visibility.quadrics.section_timeline import (
    compile_section_timeline,
)
from polyhedron_visibility.quadrics.semantic_compositing import (
    SectionCompositingAxes,
    SectionCompositingInstruction,
    SectionCompositingOverride,
    compile_section_compositing,
)
from polyhedron_visibility.quadrics.semantic_display import (
    SectionDisplayInstruction,
    SectionDisplayRole,
    compile_section_display,
)
from tikz_native.camera_3d import MultiProjectionCamera
from tikz_native.global_parallel_rig import compile_global_parallel_rig
from tikz_native.parallel_camera import ParallelCameraState
from tikz_native.parallel_preflight import (
    ParallelPreflightLimits,
    ParallelSafeFrame,
    ParallelScreenTransform,
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
        min_zoom=0.35,
        max_zoom=1.8,
    )


def _controller_options() -> dict[str, object]:
    return {
        "limits": QuadricManimLimits(
            max_surfaces=4,
            max_curves=16,
            max_fragments_per_curve=16,
            max_segments_per_fragment=160,
            max_surface_segments=512,
            max_dashes_per_fragment=128,
            max_projected_length=30.0,
            max_total_mobjects=12000,
            max_boundary_sources=24,
            max_boundary_styles=24,
        ),
        "style": QuadricManimStyle(
            surface_fill_color="#2457A7",
            surface_fill_opacity=0.32,
            surface_stroke_color="#70A5F4",
            surface_stroke_width=1.8,
            visible_curve_color="#FFE066",
            visible_curve_width=4.0,
            hidden_curve_color="#F5C84C",
            hidden_curve_width=2.3,
            hidden_curve_opacity=0.72,
            point_color="#FF5F7A",
            point_radius=0.075,
            section_plane_fill_color="#58C7B4",
            section_plane_fill_opacity=0.18,
            section_plane_stroke_color="#9AF0DE",
            section_plane_stroke_width=2.0,
            section_plane_stroke_opacity=0.90,
        ),
    }


def _compile_sphere_binding(
    scene: ThreeDScene,
    *,
    prefix: str,
    center_z: float,
    radius: float,
    final_height: float,
    initial_camera: ParallelCameraState,
    shots: ParallelCameraShotSequence,
    include_plane: bool,
    screen_transforms: Sequence[ParallelScreenTransform] | None = None,
    compositing_frames: Sequence[object] | None = None,
) -> tuple[ParallelSectionRigBinding, object]:
    surface = SphereSpec(
        f"{prefix}-sphere",
        (0.0, 0.0, center_z),
        radius,
    )
    initial_height = center_z - 0.28 * radius
    plane = SectionPlane(
        f"{prefix}-plane",
        (0.0, 0.0, initial_height),
        (0.0, 0.0, 1.0),
        u_axis=(1.0, 0.0, 0.0),
    )
    timeline = compile_section_timeline(
        f"{prefix}-section",
        surface,
        (
            ParallelPlaneTranslation(
                f"{prefix}-plane-motion",
                plane,
                (0.0, 0.0, final_height - initial_height),
                start_time=0.0,
                end_time=2.0,
            ),
        ),
    )
    banks = (f"{prefix}-bank-a", f"{prefix}-bank-b")
    catalog = build_parallel_section_rig_display_catalog(
        timeline,
        banks,
        include_plane=include_plane,
        surface_boundary_mode="certified",
    )
    display = compile_section_display(
        catalog,
        SectionDisplayInstruction.for_mode("painted"),
    )
    binding = compile_parallel_section_rig_from_shots(
        scene,
        timeline,
        shots,
        initial_camera,
        tuple(display for _ in timeline.samples),
        compositing_frames=compositing_frames,
        limits=_preflight_limits(),
        semantic_bank_ids=banks,
        frame_rate=float(config.frame_rate),
        plane_patch_margin=(0.14 if include_plane else None),
        screen_transforms=screen_transforms,
        controller_options=_controller_options(),
    )
    return binding, catalog


class _AdvancedParallelScene(ThreeDScene):
    def __init__(self, **kwargs) -> None:
        super().__init__(camera_class=MultiProjectionCamera, **kwargs)

    def _play_binding(
        self,
        binding: ParallelSectionRigBinding,
        shots: ParallelCameraShotSequence,
        *,
        acceptance: Callable[[ParallelSectionRigBinding], None] | None = None,
    ) -> None:
        coordinator = None
        try:
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
            self.wait(0.4)
        finally:
            try:
                if coordinator is not None and coordinator.active:
                    coordinator.restore()
            finally:
                binding.restore()


class ParallelViewportTangentPointDemo(_AdvancedParallelScene):
    """Use a non-identity viewport while reaching a true tangent point."""

    def construct(self) -> None:
        initial = ParallelCameraState.from_view_direction(
            (0.0, 0.0, 1.0),
            target=(0.0, 0.0, 0.25),
            screen_anchor=(0.0, -0.08),
            zoom=0.88,
        )
        shots = ParallelCameraShotSequence(
            (
                ParallelCameraShot(
                    "tangent-point-hold",
                    initial,
                    duration=2.0,
                ),
            )
        )
        transform = ParallelScreenTransform(
            inherited_zoom=1.06,
            frame_center=(-0.22, 0.14),
            display_offset=(0.12, -0.04),
        )
        transforms = (transform, transform, transform)
        self.camera.set_parallel_state(initial)
        binding, _catalog = _compile_sphere_binding(
            self,
            prefix="tangent",
            center_z=0.0,
            radius=1.25,
            final_height=1.25,
            initial_camera=initial,
            shots=shots,
            include_plane=False,
            screen_transforms=transforms,
        )
        fixed_identities = binding.controller.slot_identities()

        def require_live_tangent_point(value: ParallelSectionRigBinding) -> None:
            prepared = value.controller._last_prepared_frame
            if prepared is None or len(prepared.numeric.points) != 1:
                raise RuntimeError("tangent demo did not commit one point marker")
            point = prepared.numeric.points[0]
            if not point.visible or point.point_id not in value.allocated_point_ids:
                raise RuntimeError("tangent point lacks certified visible evidence")
            slot = value.controller._point_slots[point.point_id]
            if float(slot.get_fill_opacity()) <= 0.0:
                raise RuntimeError("tangent point slot is not visibly painted")
            if value.controller.slot_identities() != fixed_identities:
                raise RuntimeError("tangent point replaced a fixed Manim slot")
            expected = value.sequence.screen_transforms[-1]
            if self.camera.get_zoom() != expected.inherited_zoom:
                raise RuntimeError("viewport demo did not commit inherited zoom")
            if tuple(float(item) for item in self.camera.frame_center[:2]) != (
                expected.frame_center
            ):
                raise RuntimeError("viewport demo did not commit frame center")
            if value.controller.display_offset != expected.display_offset:
                raise RuntimeError("viewport demo did not commit display offset")

        self._play_binding(
            binding,
            shots,
            acceptance=require_live_tangent_point,
        )


class ParallelCompositingAxesDemo(_AdvancedParallelScene):
    """Switch opacity, occlusion participation, and depth policy separately."""

    def construct(self) -> None:
        camera = ParallelCameraState.from_view_direction(
            (0.55, 0.75, 1.0),
            target=(0.0, 0.0, 0.0),
            zoom=0.84,
        )
        shots = ParallelCameraShotSequence(
            (ParallelCameraShot("compositing-hold", camera, duration=2.0),)
        )

        # First compile only far enough to obtain the immutable catalog IDs.
        surface = SphereSpec("axes-sphere", (0.0, 0.0, 0.0), 1.25)
        plane = SectionPlane(
            "axes-plane",
            (0.0, 0.0, -0.35),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        middle_plane = SectionPlane(
            "axes-plane",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        timeline = compile_section_timeline(
            "axes-section",
            surface,
            (
                ParallelPlaneTranslation(
                    "axes-plane-motion-a",
                    plane,
                    (0.0, 0.0, 0.35),
                    start_time=0.0,
                    end_time=1.0,
                ),
                ParallelPlaneTranslation(
                    "axes-plane-motion-b",
                    middle_plane,
                    (0.0, 0.0, 0.35),
                    start_time=1.0,
                    end_time=2.0,
                ),
            ),
        )
        banks = ("axes-bank-a", "axes-bank-b")
        catalog = build_parallel_section_rig_display_catalog(
            timeline,
            banks,
            include_plane=False,
            surface_boundary_mode="certified",
        )
        display = compile_section_display(
            catalog,
            SectionDisplayInstruction.for_mode("painted"),
        )
        surface_fill_slot = next(
            slot.slot_id
            for slot in catalog.slots
            if slot.role is SectionDisplayRole.SURFACE_FILL
        )
        def compositing_state(
            *,
            opacity: float,
            participation: str,
            depth: str,
        ):
            return compile_section_compositing(
                catalog,
                SectionCompositingInstruction.for_catalog(
                    catalog,
                    defaults=SectionCompositingAxes(
                        depth_presentation=depth,
                    ),
                    overrides=(
                        SectionCompositingOverride.for_slot(
                            surface_fill_slot,
                            display_opacity=opacity,
                            occlusion_participation=participation,
                        ),
                    ),
                ),
            )

        authored_compositing = (
            compositing_state(
                opacity=1.0,
                participation="certified",
                depth="diagrammatic",
            ),
            compositing_state(
                opacity=0.0,
                participation="certified",
                depth="diagrammatic",
            ),
            compositing_state(
                opacity=0.0,
                participation="paint-only",
                depth="diagrammatic",
            ),
            compositing_state(
                opacity=0.0,
                participation="paint-only",
                depth="physical",
            ),
            compositing_state(
                opacity=1.0,
                participation="paint-only",
                depth="physical",
            ),
        )
        self.camera.set_parallel_state(camera)
        binding = compile_parallel_section_rig_from_shots(
            self,
            timeline,
            shots,
            camera,
            tuple(display for _ in timeline.samples),
            compositing_frames=authored_compositing,
            limits=_preflight_limits(),
            semantic_bank_ids=banks,
            frame_rate=float(config.frame_rate),
            plane_patch_margin=None,
            controller_options=_controller_options(),
        )
        surface_id = surface.surface_id

        def require_independent_axes(value: ParallelSectionRigBinding) -> None:
            resolved = []
            for frame in authored_compositing:
                slot = next(
                    item for item in frame.slots
                    if item.slot_id == surface_fill_slot
                )
                resolved.append(
                    (
                        slot.display_opacity,
                        slot.occlusion_participation.value,
                        slot.depth_presentation.value,
                    )
                )
            for left, right in zip(resolved, resolved[1:]):
                if sum(a != b for a, b in zip(left, right)) != 1:
                    raise RuntimeError(
                        "compositing demo did not change exactly one axis per step"
                    )
            if value._compositing_frame.digest != authored_compositing[-1].digest:
                raise RuntimeError("final compositing frame was not committed")
            if value._resolve_surface_opacities()[surface_id] != 1.0:
                raise RuntimeError("final surface display opacity is not visible")
            if value._resolve_occluding_surface_ids():
                raise RuntimeError("final paint-only surface still occludes")
            if value._resolve_paint_policy().value != "physical":
                raise RuntimeError("final physical depth policy was not committed")
            prepared = value.controller._last_prepared_frame
            if (
                prepared is None
                or prepared.numeric.surface_opacities.get(surface_id) != 1.0
                or value.controller.last_frame is None
                or value.controller.last_frame.paint_policy.value != "physical"
            ):
                raise RuntimeError("live controller evidence differs from final axes")

        self._play_binding(
            binding,
            shots,
            acceptance=require_independent_axes,
        )


class GlobalParallelRigOcclusionDemo(_AdvancedParallelScene):
    """Use one global painter for two projected-overlapping sphere Rigs."""

    def construct(self) -> None:
        camera = ParallelCameraState.from_view_direction(
            (0.0, 0.0, 1.0),
            zoom=0.82,
        )
        shots = ParallelCameraShotSequence(
            (ParallelCameraShot("global-hold", camera, duration=2.0),)
        )
        self.camera.set_parallel_state(camera)
        far, _far_catalog = _compile_sphere_binding(
            self,
            prefix="global-far",
            center_z=0.0,
            radius=0.82,
            final_height=0.82,
            initial_camera=camera,
            shots=shots,
            include_plane=False,
        )
        near, _near_catalog = _compile_sphere_binding(
            self,
            prefix="global-near",
            center_z=3.0,
            radius=1.42,
            final_height=3.38,
            initial_camera=camera,
            shots=shots,
            include_plane=False,
        )
        global_binding = compile_global_parallel_rig((far, near))
        coordinator = None
        driver = VMobject()
        clock = ValueTracker(0.0)
        latest = {"index": -1}
        try:
            global_binding.attach()
            coordinator = global_binding.build_coordinator(self.camera)
            frames = global_binding.sequence.frames
            coordinator.update(frames[0])
            latest["index"] = 0

            def update_global_frame(_mobject: VMobject) -> None:
                index = min(
                    len(frames) - 1,
                    max(0, int(round(clock.get_value()))),
                )
                if index != latest["index"]:
                    coordinator.update(frames[index])
                    latest["index"] = index

            driver.add_updater(update_global_frame)
            self.add(driver)
            self.play(
                clock.animate.set_value(float(len(frames) - 1)),
                run_time=2.0,
                rate_func=linear,
            )
            coordinator.update(frames[-1])
            latest["index"] = len(frames) - 1
            self.wait(0.4)
            prepared = global_binding.controller._last_prepared_frame
            if prepared is None or not any(
                point.point_id.startswith("global-far-") and not point.visible
                for point in prepared.numeric.points
            ):
                raise RuntimeError(
                    "global demo did not certify the hidden far tangent point"
                )
        finally:
            driver.clear_updaters()
            self.remove(driver)
            try:
                if coordinator is not None and coordinator.active:
                    coordinator.restore()
            finally:
                global_binding.restore()


__all__ = [
    "GlobalParallelRigOcclusionDemo",
    "ParallelCompositingAxesDemo",
    "ParallelViewportTangentPointDemo",
]
