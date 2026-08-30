"""One-controller Manim binding for certified parallel section sequences.

The renderer-neutral :mod:`tikz_native.quadric_section_parallel` contract owns
camera sampling, topology banks, finite plane patches, semantic display state,
and preflight evidence.  This module is the deliberately small Cairo seam that
loads those channels into one :class:`QuadricOcclusion3D` controller.

Two details are important:

* both topology banks share one surface controller and one painter band;
* painter evidence is obtained from the controller's real numeric preparation,
  then compiled back into the final source-authoritative sequence.

No Scene object is attached while painter evidence is being compiled.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from polyhedron_visibility.quadrics.animation import (
    _materialize_tracked_section_curves,
    match_tracked_section_frame,
)
from polyhedron_visibility.quadrics.critical import AnalyticCurve3D
from polyhedron_visibility.quadrics.curves import (
    ParametricConicBranch,
    SegmentCurve,
)
from polyhedron_visibility.topology import ParameterInterval
from polyhedron_visibility.quadrics.manim import (
    QuadricManimStyle,
    QuadricOcclusion3D,
    QuadricOcclusionTransactionSnapshot,
)
from polyhedron_visibility.quadrics.plane_patch import FittedPlaneDisplayPatch
from polyhedron_visibility.quadrics.section_timeline import SectionTimeline
from polyhedron_visibility.quadrics.section_timeline_transition import (
    SectionTimelineTransitionMode,
)
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section_boundary,
)
from polyhedron_visibility.quadrics.semantic_display import (
    SectionDisplayCatalog,
    SectionDisplayFrame,
    SectionDisplayRole,
    SectionSemanticSlot,
)

from .parallel_camera import ParallelCameraState
from .parallel_frame import (
    ParallelFrameCoordinator,
    parallel_camera_frame_participant,
)
from .parallel_preflight import (
    PainterOrderEvidence,
    ParallelPreflightLimits,
    ParallelScreenTransform,
)
from .parallel_shots import ParallelCameraShotSequence
from .quadric_section_parallel import (
    SECTION_PLANE_CHANNEL,
    SECTION_TRANSITION_STATE_CHANNEL,
    ParallelSectionSequence,
    _bank_render_frame,
    _timeline_plane_at_time,
    compile_parallel_section_sequence_from_shots,
    parallel_screen_transform_guard,
    parallel_section_preflight_gate,
    section_display_frame_participant,
    section_painter_order_participant,
    section_plane_patch_participant,
)
from .section_bank_render import (
    SectionBankRenderFrame,
    section_bank_frame_participant,
)


class ParallelSectionRigBindingError(RuntimeError):
    """A certified section sequence cannot be represented by this binding."""


_CURVE_INTERVAL_CAPACITY = 2
_IDENTITY_SCREEN_TRANSFORM = ParallelScreenTransform()
_STATIC_PAINT_ROLES = frozenset(
    {
        SectionDisplayRole.SURFACE_FILL,
        SectionDisplayRole.SURFACE_OUTLINE,
        SectionDisplayRole.PLANE_FILL,
        SectionDisplayRole.PLANE_OUTLINE,
    }
)
_UNSUPPORTED_BOUNDARY_ROLES = frozenset(
    {
        SectionDisplayRole.GENERATOR,
        SectionDisplayRole.CONTOUR,
        SectionDisplayRole.CAP_RIM,
    }
)


def _camera_equal(left: ParallelCameraState, right: ParallelCameraState) -> bool:
    return bool(
        np.array_equal(left.matrix, right.matrix)
        and np.array_equal(left.target, right.target)
        and np.array_equal(left.screen_anchor, right.screen_anchor)
        and left.zoom == right.zoom
    )


def _painter_evidence(draw_order: Sequence[str]) -> PainterOrderEvidence:
    order = tuple(draw_order)
    if not order:
        raise ParallelSectionRigBindingError(
            "the real quadric controller produced an empty painter order"
        )
    return PainterOrderEvidence(
        item_ids=order,
        relations=tuple(zip(order, order[1:])),
        draw_order=order,
    )


def _require_semantic_scene_camera(scene: object) -> object:
    camera = getattr(scene, "camera", None)
    snapshot = getattr(camera, "snapshot_parallel_state", None)
    setter = getattr(camera, "set_parallel_state", None)
    get_zoom = getattr(camera, "get_zoom", None)
    frame_center = getattr(camera, "frame_center", None)
    if not callable(snapshot) or not callable(setter):
        raise ParallelSectionRigBindingError(
            "scene.camera must be a semantic parallel camera with snapshot and "
            "set methods"
        )
    if not callable(get_zoom) or frame_center is None:
        raise ParallelSectionRigBindingError(
            "scene.camera does not expose its inherited screen transform"
        )
    try:
        state = snapshot()
    except Exception as exc:
        raise ParallelSectionRigBindingError(
            f"scene.camera is not in a parallel snapshot state: {exc}"
        ) from exc
    if not isinstance(state, ParallelCameraState):
        raise ParallelSectionRigBindingError(
            "scene.camera snapshot did not return ParallelCameraState"
        )
    return camera


def _camera_screen_transform(
    camera: object,
    *,
    display_offset: Sequence[float] = (0.0, 0.0),
) -> ParallelScreenTransform:
    get_zoom = getattr(camera, "get_zoom")
    frame_center = np.asarray(getattr(camera, "frame_center"), dtype=float)
    if frame_center.shape != (3,) or not np.all(np.isfinite(frame_center)):
        raise ParallelSectionRigBindingError(
            "scene.camera frame_center must contain three finite values"
        )
    return ParallelScreenTransform(
        inherited_zoom=float(get_zoom()),
        frame_center=tuple(float(item) for item in frame_center[:2]),
        display_offset=tuple(float(item) for item in display_offset),
    )


def build_parallel_section_rig_display_catalog(
    timeline: SectionTimeline,
    semantic_bank_ids: tuple[str, str],
    *,
    include_plane: bool,
) -> SectionDisplayCatalog:
    """Build the complete fixed-slot catalog owned by this binding.

    Each topology bank receives two branch slots, one future point slot, and
    one cap-chord slot for every analytically possible finite cap chord.
    Isolated points are preflighted by the renderer-neutral layer, but the
    current Cairo binding still rejects a frame that activates one.
    """

    if not isinstance(timeline, SectionTimeline):
        raise TypeError("timeline must be a SectionTimeline")
    if len(semantic_bank_ids) != 2 or len(set(semantic_bank_ids)) != 2:
        raise ParallelSectionRigBindingError(
            "semantic_bank_ids must contain two unique identities"
        )
    prefix = timeline.section_id
    slots: list[SectionSemanticSlot] = [
        SectionSemanticSlot(
            f"{prefix}:display:surface-fill",
            SectionDisplayRole.SURFACE_FILL,
        ),
        SectionSemanticSlot(
            f"{prefix}:display:surface-outline",
            SectionDisplayRole.SURFACE_OUTLINE,
        ),
    ]
    if include_plane:
        slots.extend(
            (
                SectionSemanticSlot(
                    f"{prefix}:display:plane-fill",
                    SectionDisplayRole.PLANE_FILL,
                ),
                SectionSemanticSlot(
                    f"{prefix}:display:plane-outline",
                    SectionDisplayRole.PLANE_OUTLINE,
                ),
            )
        )
    for bank_id in semantic_bank_ids:
        slots.extend(
            SectionSemanticSlot(
                f"{prefix}:display:{bank_id}:curve:{slot_index}",
                SectionDisplayRole.SECTION_CURVE,
                topology_bank=bank_id,
            )
            for slot_index in range(2)
        )
        slots.append(
            SectionSemanticSlot(
                f"{prefix}:display:{bank_id}:point:0",
                SectionDisplayRole.SECTION_POINT,
                topology_bank=bank_id,
            )
        )
        slots.extend(
            SectionSemanticSlot(
                f"{prefix}:display:{bank_id}:cap:{source_id}",
                SectionDisplayRole.CAP_CHORD,
                source_id=source_id,
                topology_bank=bank_id,
            )
            for source_id in timeline.cap_chord_ids
        )
    return SectionDisplayCatalog(timeline.section_id, tuple(slots))


@dataclass(frozen=True, slots=True)
class _BankSnapshot:
    frame: SectionBankRenderFrame
    curves: tuple[AnalyticCurve3D, ...]
    base_opacities: tuple[tuple[str, float], ...]
    section_sources_authoritative: bool
    controller_section_id: str | None
    controller_section_coefficient_tolerance: float | None


@dataclass(frozen=True, slots=True)
class _DisplaySnapshot:
    frame: SectionDisplayFrame
    controller: QuadricOcclusionTransactionSnapshot


class ParallelSectionRigBinding:
    """Load one parallel section sequence into one quadric controller.

    Construct this class from a draft sequence, use
    :meth:`painter_order_provider` while compiling the final sequence, then call
    :meth:`bind_sequence`.  The convenience function
    :func:`compile_parallel_section_rig_from_shots` performs those steps.
    """

    def __init__(
        self,
        scene: object,
        draft_sequence: ParallelSectionSequence,
        *,
        controller_options: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(draft_sequence, ParallelSectionSequence):
            raise TypeError("draft_sequence must be a ParallelSectionSequence")
        self.scene = scene
        camera = _require_semantic_scene_camera(scene)
        if _camera_screen_transform(camera) != _IDENTITY_SCREEN_TRANSFORM:
            raise ParallelSectionRigBindingError(
                "the live camera screen transform must be identity before binding"
            )
        self._draft_sequence = draft_sequence
        self._sequence: ParallelSectionSequence | None = None
        self._coordinator: ParallelFrameCoordinator | None = None
        self._validate_sequence_contract(draft_sequence)

        first_display = draft_sequence.display_frames[0]
        self._curve_slots_by_bank = {
            bank_id: tuple(
                sorted(
                    item.slot_id
                    for item in first_display.slots
                    if item.role is SectionDisplayRole.SECTION_CURVE
                    and item.topology_bank == bank_id
                )
            )
            for bank_id in draft_sequence.semantic_bank_ids
        }
        self._cap_slot_by_bank_source = {
            (item.topology_bank, item.source_id): item.slot_id
            for item in first_display.slots
            if item.role is SectionDisplayRole.CAP_CHORD
        }
        self._physical_to_semantic: dict[str, str] = {}
        physical_ids: list[str] = []
        for bank_id in draft_sequence.semantic_bank_ids:
            for semantic_id in self._curve_slots_by_bank[bank_id]:
                for interval_index in range(_CURVE_INTERVAL_CAPACITY):
                    physical_id = f"{semantic_id}:interval:{interval_index}"
                    physical_ids.append(physical_id)
                    self._physical_to_semantic[physical_id] = semantic_id
        for semantic_id in self._cap_slot_by_bank_source.values():
            physical_ids.append(semantic_id)
            self._physical_to_semantic[semantic_id] = semantic_id
        if len(set(physical_ids)) != len(physical_ids):
            raise ParallelSectionRigBindingError(
                "semantic curve slots produce colliding physical curve identities"
            )
        self._allocated_curve_ids = tuple(sorted(physical_ids))

        self._bank_frame = draft_sequence.bank_render_frames[0]
        self._display_frame = draft_sequence.display_frames[0]
        self._patch_fit = draft_sequence.plane_patch_fits[0]
        self._painter_order = draft_sequence.painter_orders[0]
        self._curves: tuple[AnalyticCurve3D, ...] = ()
        self._base_curve_opacities: dict[str, float] = {}
        self._projection_override: ParallelCameraState | None = (
            draft_sequence.camera_samples[0].state
        )
        self._dry_painters: dict[float, PainterOrderEvidence] = {}
        self._section_sources_authoritative = False
        self._apply_bank_geometry(self._bank_frame)

        options = dict(controller_options or {})
        style = options.pop("style", QuadricManimStyle())
        if not isinstance(style, QuadricManimStyle):
            raise TypeError(
                "controller_options['style'] must be a QuadricManimStyle"
            )
        static_opacities = {
            item.role: item.opacity_multiplier
            for item in first_display.slots
            if item.role in _STATIC_PAINT_ROLES
        }
        style = replace(
            style,
            surface_fill_opacity=(
                style.surface_fill_opacity
                * static_opacities[SectionDisplayRole.SURFACE_FILL]
            ),
            surface_stroke_opacity=(
                style.surface_stroke_opacity
                * static_opacities[SectionDisplayRole.SURFACE_OUTLINE]
            ),
            section_plane_fill_opacity=(
                style.section_plane_fill_opacity
                * static_opacities.get(SectionDisplayRole.PLANE_FILL, 1.0)
            ),
            section_plane_stroke_opacity=(
                style.section_plane_stroke_opacity
                * static_opacities.get(SectionDisplayRole.PLANE_OUTLINE, 1.0)
            ),
        )
        options["style"] = style
        owned = {
            "surfaces",
            "curves",
            "projection",
            "allocated_curve_ids",
            "curve_opacities",
            "section_id",
            "section_coefficient_tolerance",
            "section_plane",
            "section_patch",
            "context",
            "surface_order_mode",
            "boundary_visibility_mode",
            "include_surface_boundaries",
            "display_offset",
            "automatic_updates",
            "legacy_surface_stroke_fallback",
        }
        conflicts = tuple(sorted(owned & set(options)))
        if conflicts:
            raise ParallelSectionRigBindingError(
                "controller_options cannot override binding-owned keys: "
                + ", ".join(conflicts)
            )
        surface = draft_sequence.timeline.samples[0].surface
        section_enabled = self._patch_fit is not None
        self.controller = QuadricOcclusion3D(
            scene,
            surfaces=(surface,),
            curves=self._resolve_curves,
            projection=self._resolve_projection,
            allocated_curve_ids=self._allocated_curve_ids,
            curve_opacities=self._resolve_curve_opacities,
            section_id=(
                draft_sequence.timeline.section_id
                if self._section_sources_authoritative and section_enabled
                else None
            ),
            section_coefficient_tolerance=(
                draft_sequence.timeline.coefficient_tolerance
                if self._section_sources_authoritative and section_enabled
                else None
            ),
            section_plane=(self._resolve_plane if section_enabled else None),
            section_patch=(self._resolve_patch if section_enabled else None),
            context=draft_sequence.timeline.geometry_context,
            surface_order_mode="automatic",
            boundary_visibility_mode="unified",
            include_surface_boundaries=False,
            display_offset=(0.0, 0.0),
            automatic_updates=False,
            legacy_surface_stroke_fallback=True,
            **options,
        )

    @property
    def sequence(self) -> ParallelSectionSequence:
        if self._sequence is None:
            raise ParallelSectionRigBindingError(
                "the final sequence has not been bound yet"
            )
        return self._sequence

    @property
    def allocated_curve_ids(self) -> tuple[str, ...]:
        return self._allocated_curve_ids

    @property
    def attached(self) -> bool:
        return self.controller.attached

    @property
    def display_mobject(self):
        return self.controller.display_mobject

    def _validate_sequence_contract(self, sequence: ParallelSectionSequence) -> None:
        if any(item != _IDENTITY_SCREEN_TRANSFORM for item in sequence.screen_transforms):
            raise ParallelSectionRigBindingError(
                "the first Cairo binding supports only identity screen transforms; "
                "put target, screen anchor, and zoom in ParallelCameraState"
            )
        if any(
            layer.isolated_point_count
            for frame in sequence.bank_render_frames
            for layer in frame.layers
        ):
            raise ParallelSectionRigBindingError(
                "SECTION_POINT activation requires a true fixed Manim point slot; "
                "this binding refuses to drop or imitate isolated points"
            )
        first = sequence.display_frames[0]
        first_metadata = tuple(
            (item.slot_id, item.role, item.source_id, item.topology_bank)
            for item in first.slots
        )
        first_static_opacities = {
            item.slot_id: item.opacity_multiplier
            for item in first.slots
            if item.role in _STATIC_PAINT_ROLES
        }
        for frame in sequence.display_frames:
            metadata = tuple(
                (item.slot_id, item.role, item.source_id, item.topology_bank)
                for item in frame.slots
            )
            if frame.section_id != sequence.timeline.section_id:
                raise ParallelSectionRigBindingError(
                    "display frame section identity differs from the timeline"
                )
            if metadata != first_metadata:
                raise ParallelSectionRigBindingError(
                    "semantic display slot metadata changed between frames"
                )
            unsupported = tuple(
                item.role.value
                for item in frame.slots
                if item.role in _UNSUPPORTED_BOUNDARY_ROLES
            )
            if unsupported:
                raise ParallelSectionRigBindingError(
                    "the first binding does not own generator/contour/cap-rim "
                    "semantic slots: " + ", ".join(sorted(set(unsupported)))
                )
            for item in frame.slots:
                if (
                    item.role in _STATIC_PAINT_ROLES
                    and item.opacity_multiplier
                    != first_static_opacities[item.slot_id]
                ):
                    raise ParallelSectionRigBindingError(
                        f"opacity for static display role {item.role.value!r} "
                        "must remain constant across the sequence"
                    )
        required_roles = {
            SectionDisplayRole.SURFACE_FILL,
            SectionDisplayRole.SURFACE_OUTLINE,
        }
        if sequence.plane_patch_margin is not None:
            required_roles.update(
                {
                    SectionDisplayRole.PLANE_FILL,
                    SectionDisplayRole.PLANE_OUTLINE,
                }
            )
        for role in required_roles:
            count = sum(item.role is role for item in first.slots)
            if count != 1:
                raise ParallelSectionRigBindingError(
                    f"the binding requires exactly one {role.value!r} slot"
                )
        for bank_id in sequence.semantic_bank_ids:
            curve_slots = tuple(
                item
                for item in first.slots
                if item.role is SectionDisplayRole.SECTION_CURVE
                and item.topology_bank == bank_id
            )
            if len(curve_slots) < 2:
                raise ParallelSectionRigBindingError(
                    f"topology bank {bank_id!r} requires two section-curve slots"
                )

    def _sequence_structure_equal(
        self,
        left: ParallelSectionSequence,
        right: ParallelSectionSequence,
    ) -> bool:
        cameras_equal = len(left.camera_samples) == len(right.camera_samples) and all(
            left_sample.sample_id == right_sample.sample_id
            and left_sample.time == right_sample.time
            and left_sample.shot_id == right_sample.shot_id
            and left_sample.phase is right_sample.phase
            and _camera_equal(left_sample.state, right_sample.state)
            for left_sample, right_sample in zip(
                left.camera_samples,
                right.camera_samples,
            )
        )
        left_provenance = (
            None
            if left.camera_provenance is None
            else left.camera_provenance.to_dict()
        )
        right_provenance = (
            None
            if right.camera_provenance is None
            else right.camera_provenance.to_dict()
        )
        left_framing = tuple(
            item.framing_points for item in left.preflight_frames
        )
        right_framing = tuple(
            item.framing_points for item in right.preflight_frames
        )
        return bool(
            left.timeline is right.timeline
            and left.schema == right.schema
            and left.transition_plan.to_dict() == right.transition_plan.to_dict()
            and left.evaluation_times == right.evaluation_times
            and left.semantic_bank_ids == right.semantic_bank_ids
            and cameras_equal
            and left.display_frames == right.display_frames
            and left.bank_render_frames == right.bank_render_frames
            and left.plane_patch_margin == right.plane_patch_margin
            and left.plane_patch_fits == right.plane_patch_fits
            and left.screen_transforms == right.screen_transforms
            and left.preflight_limits == right.preflight_limits
            and left_framing == right_framing
            and left_provenance == right_provenance
        )

    def _frame_index(self, time: float) -> int:
        try:
            return self._draft_sequence.evaluation_times.index(float(time))
        except ValueError as exc:
            raise ParallelSectionRigBindingError(
                f"painter provider received unauthored time {time!r}"
            ) from exc

    def _resolve_curves(self) -> tuple[AnalyticCurve3D, ...]:
        return self._curves

    def _resolve_curve_opacities(self) -> Mapping[str, float]:
        display_opacity = {
            item.slot_id: item.opacity_multiplier
            for item in self._display_frame.slots
        }
        return {
            physical_id: base_opacity
            * display_opacity[self._physical_to_semantic[physical_id]]
            for physical_id, base_opacity in self._base_curve_opacities.items()
        }

    def _resolve_projection(self, _scene: object) -> ParallelCameraState:
        if self._projection_override is not None:
            return self._projection_override
        snapshot = getattr(self.scene.camera, "snapshot_parallel_state", None)
        if not callable(snapshot):
            raise ParallelSectionRigBindingError(
                "scene.camera must provide snapshot_parallel_state()"
            )
        state = snapshot()
        if not isinstance(state, ParallelCameraState):
            raise ParallelSectionRigBindingError(
                "scene.camera returned an invalid parallel camera state"
            )
        return state

    def _resolve_plane(self):
        if self._patch_fit is None:
            raise ParallelSectionRigBindingError("the staged frame has no plane patch")
        return self._patch_fit.plane

    def _resolve_patch(self):
        if self._patch_fit is None:
            raise ParallelSectionRigBindingError("the staged frame has no plane patch")
        return self._patch_fit.patch

    def _apply_bank_geometry(self, frame: SectionBankRenderFrame) -> None:
        timeline = self._draft_sequence.timeline
        index = self._frame_index(frame.time)
        transition = self._draft_sequence.frames[index].channel(
            SECTION_TRANSITION_STATE_CHANNEL
        )
        expected = _bank_render_frame(
            timeline,
            transition,
            self._draft_sequence.semantic_bank_ids,
        )
        if frame != expected:
            raise ParallelSectionRigBindingError(
                "bank frame or geometry digest differs from the source timeline"
            )
        self._section_sources_authoritative = all(
            layer.geometry_time == frame.time for layer in frame.layers
        )
        curves: list[AnalyticCurve3D] = []
        base_opacities: dict[str, float] = {}
        for layer in frame.layers:
            plane = _timeline_plane_at_time(timeline, layer.geometry_time)
            boundary = compute_quadric_section_boundary(
                timeline.section_id,
                timeline.samples[0].surface,
                plane,
                context=timeline.geometry_context,
                coefficient_tolerance=timeline.coefficient_tolerance,
            )
            reference = timeline.animation.frames[layer.reference_frame_index]
            tracked = match_tracked_section_frame(
                reference,
                boundary.trace,
                frame_index=reference.frame_index,
                time=layer.geometry_time,
            )
            bank_slots = self._curve_slots_by_bank[layer.semantic_bank_id]

            def curve_id(mapping, interval_index: int) -> str:
                return (
                    f"{bank_slots[mapping.capacity_slot]}:interval:"
                    f"{interval_index}"
                )

            materialized = _materialize_tracked_section_curves(
                tracked,
                curve_id,
                max_intervals_per_component=_CURVE_INTERVAL_CAPACITY,
            )
            split_materialized: list[ParametricConicBranch] = []
            for curve in materialized:
                if not curve.closed:
                    split_materialized.append(curve)
                    continue
                midpoint = curve.domain.start + 0.5 * curve.domain.length
                prefix, marker, _index = curve.curve_id.rpartition(":interval:")
                if not marker:
                    raise ParallelSectionRigBindingError(
                        "closed branch lacks its fixed interval identity"
                    )
                split_materialized.extend(
                    (
                        ParametricConicBranch(
                            f"{prefix}:interval:0",
                            curve.parameterization,
                            curve.plane_embedding,
                            ParameterInterval(curve.domain.start, midpoint),
                        ),
                        ParametricConicBranch(
                            f"{prefix}:interval:1",
                            curve.parameterization,
                            curve.plane_embedding,
                            ParameterInterval(midpoint, curve.domain.end),
                        ),
                    )
                )
            materialized = tuple(split_materialized)
            if len(tracked.branches) != layer.branch_count:
                raise ParallelSectionRigBindingError(
                    "materialized branch count differs from bank evidence"
                )
            for curve in materialized:
                curves.append(curve)
                base_opacities[curve.curve_id] = layer.opacity

            chord_map = {item.curve_id: item for item in boundary.cap_chords}
            if set(chord_map) != set(layer.active_cap_chord_ids):
                raise ParallelSectionRigBindingError(
                    "materialized cap chords differ from bank evidence"
                )
            for source_id in layer.active_cap_chord_ids:
                try:
                    semantic_id = self._cap_slot_by_bank_source[
                        (layer.semantic_bank_id, source_id)
                    ]
                except KeyError as exc:
                    raise ParallelSectionRigBindingError(
                        "active cap chord has no banked semantic slot"
                    ) from exc
                chord = chord_map[source_id]
                curves.append(
                    SegmentCurve(
                        semantic_id,
                        chord.start,
                        chord.end,
                        chord.domain,
                    )
                )
                base_opacities[semantic_id] = layer.opacity
        identities = tuple(item.curve_id for item in curves)
        if len(set(identities)) != len(identities):
            raise ParallelSectionRigBindingError(
                "active topology banks produced duplicate physical curve ids"
            )
        self._curves = tuple(sorted(curves, key=lambda item: item.curve_id))
        self._base_curve_opacities = base_opacities
        controller = getattr(self, "controller", None)
        if controller is not None:
            controller.section_id = (
                timeline.section_id
                if self._section_sources_authoritative
                and self._patch_fit is not None
                else None
            )
            controller.section_coefficient_tolerance = (
                timeline.coefficient_tolerance
                if self._section_sources_authoritative
                and self._patch_fit is not None
                else None
            )

    def painter_order_provider(
        self,
        time: float,
        camera: ParallelCameraState,
        plane,
    ) -> PainterOrderEvidence:
        """Prepare the real controller numerics without attaching Scene objects."""

        if self.controller.attached:
            raise ParallelSectionRigBindingError(
                "painter evidence cannot be recompiled after Scene attachment"
            )
        index = self._frame_index(time)
        expected_camera = self._draft_sequence.camera_samples[index].state
        expected_patch = self._draft_sequence.plane_patch_fits[index]
        expected_plane = (
            self._draft_sequence.frames[index].channel(SECTION_PLANE_CHANNEL)
        )
        if not _camera_equal(camera, expected_camera) or plane != expected_plane:
            raise ParallelSectionRigBindingError(
                "painter provider inputs differ from the draft sequence"
            )
        self._projection_override = camera
        self._bank_frame = self._draft_sequence.bank_render_frames[index]
        self._display_frame = self._draft_sequence.display_frames[index]
        self._patch_fit = expected_patch
        self._apply_bank_geometry(self._bank_frame)
        numeric = self.controller._prepare_numeric(None)
        evidence = _painter_evidence(numeric.painter_draw_order)
        self._dry_painters[float(time)] = evidence
        return evidence

    def bind_sequence(self, sequence: ParallelSectionSequence) -> None:
        if not isinstance(sequence, ParallelSectionSequence):
            raise TypeError("sequence must be a ParallelSectionSequence")
        if self.controller.attached:
            raise ParallelSectionRigBindingError(
                "a final sequence cannot be rebound after Scene attachment"
            )
        self._validate_sequence_contract(sequence)
        if not self._sequence_structure_equal(self._draft_sequence, sequence):
            raise ParallelSectionRigBindingError(
                "final sequence changed non-painter inputs from the draft"
            )
        expected = tuple(
            self._dry_painters.get(time) for time in sequence.evaluation_times
        )
        if any(item is None for item in expected) or tuple(expected) != (
            sequence.painter_orders
        ):
            raise ParallelSectionRigBindingError(
                "final sequence was not compiled with this binding's real painter provider"
            )
        self._sequence = sequence
        self._reset_to_first_frame()

    def _reset_to_first_frame(self) -> None:
        sequence = self.sequence
        self._projection_override = sequence.camera_samples[0].state
        self._bank_frame = sequence.bank_render_frames[0]
        self._display_frame = sequence.display_frames[0]
        self._patch_fit = sequence.plane_patch_fits[0]
        self._painter_order = sequence.painter_orders[0]
        self._apply_bank_geometry(self._bank_frame)

    def attach(self) -> "ParallelSectionRigBinding":
        sequence = self.sequence
        if self.controller.attached:
            return self
        if self._live_screen_transform() != _IDENTITY_SCREEN_TRANSFORM:
            raise ParallelSectionRigBindingError(
                "the live renderer screen transform differs from preflight"
            )
        self._reset_to_first_frame()
        try:
            self.controller.attach()
            if self._actual_painter_order() != sequence.painter_orders[0].draw_order:
                raise ParallelSectionRigBindingError(
                    "attached controller painter order differs from preflight evidence"
                )
        except BaseException:
            if self.controller.attached:
                self.controller.restore()
            raise
        finally:
            self._projection_override = None
        return self

    def restore(self) -> "ParallelSectionRigBinding":
        if self._coordinator is not None and self._coordinator.active:
            self._coordinator.restore()
        if self.controller.attached:
            self.controller.restore()
        if self._sequence is not None:
            self._reset_to_first_frame()
        return self

    def snapshot_section_bank_render_state(self) -> _BankSnapshot:
        return _BankSnapshot(
            self._bank_frame,
            self._curves,
            tuple(sorted(self._base_curve_opacities.items())),
            self._section_sources_authoritative,
            self.controller.section_id,
            self.controller.section_coefficient_tolerance,
        )

    def apply_section_bank_render_frame(self, frame: SectionBankRenderFrame) -> None:
        if not isinstance(frame, SectionBankRenderFrame):
            raise TypeError("frame must be a SectionBankRenderFrame")
        self._bank_frame = frame
        self._apply_bank_geometry(frame)

    def restore_section_bank_render_state(self, snapshot: _BankSnapshot) -> None:
        if not isinstance(snapshot, _BankSnapshot):
            raise TypeError("snapshot must be a bank snapshot")
        self._bank_frame = snapshot.frame
        self._curves = snapshot.curves
        self._base_curve_opacities = dict(snapshot.base_opacities)
        self._section_sources_authoritative = (
            snapshot.section_sources_authoritative
        )
        self.controller.section_id = snapshot.controller_section_id
        self.controller.section_coefficient_tolerance = (
            snapshot.controller_section_coefficient_tolerance
        )

    def snapshot_section_plane_patch_state(self) -> FittedPlaneDisplayPatch | None:
        return self._patch_fit

    def apply_section_plane_patch_fit(
        self,
        fit: FittedPlaneDisplayPatch | None,
    ) -> None:
        if fit is not None and not isinstance(fit, FittedPlaneDisplayPatch):
            raise TypeError("fit must be FittedPlaneDisplayPatch or None")
        if (fit is None) != (self._draft_sequence.plane_patch_margin is None):
            raise ParallelSectionRigBindingError(
                "plane patch activation cannot change after controller allocation"
            )
        self._patch_fit = fit

    def restore_section_plane_patch_state(
        self,
        fit: FittedPlaneDisplayPatch | None,
    ) -> None:
        self._patch_fit = fit

    def snapshot_section_painter_order_state(self) -> PainterOrderEvidence:
        return self._painter_order

    def apply_section_painter_order(self, value: PainterOrderEvidence) -> None:
        if not isinstance(value, PainterOrderEvidence):
            raise TypeError("value must be PainterOrderEvidence")
        self._painter_order = value

    def restore_section_painter_order_state(self, value: PainterOrderEvidence) -> None:
        if not isinstance(value, PainterOrderEvidence):
            raise TypeError("value must be PainterOrderEvidence")
        self._painter_order = value

    def snapshot_section_display_state(self) -> _DisplaySnapshot:
        if not self.controller.attached:
            raise ParallelSectionRigBindingError(
                "display transactions require an attached controller"
            )
        return _DisplaySnapshot(
            self._display_frame,
            self.controller.snapshot_transaction_state(),
        )

    def _actual_painter_order(self) -> tuple[str, ...]:
        expected_z = self.controller.active_painter_z_indices
        prepared = self.controller._last_prepared_frame
        if prepared is None:
            raise ParallelSectionRigBindingError(
                "controller has no committed painter frame"
            )
        item_mobjects = prepared.numeric.item_mobjects
        if set(item_mobjects) != set(expected_z):
            raise ParallelSectionRigBindingError(
                "controller painter cache differs from the committed item set"
            )
        signature = {
            item_id: (root_id, z_index)
            for item_id, root_id, z_index
            in self.controller._last_painter_band_signature
        }
        live: list[tuple[str, float]] = []
        for item_id, root in item_mobjects.items():
            if item_id not in signature or signature[item_id][0] != id(root):
                raise ParallelSectionRigBindingError(
                    "controller painter item identity drifted after preflight"
                )
            live_z = float(getattr(root, "z_index", float("nan")))
            if not np.isfinite(live_z) or live_z != expected_z[item_id]:
                raise ParallelSectionRigBindingError(
                    "controller live painter z-index differs from its managed band"
                )
            if signature[item_id][1] != live_z:
                raise ParallelSectionRigBindingError(
                    "controller painter signature differs from live z-index"
                )
            live.append((item_id, live_z))
        if len({value for _item_id, value in live}) != len(live):
            raise ParallelSectionRigBindingError(
                "controller live painter band contains duplicate z indices"
            )
        return tuple(item_id for item_id, _z in sorted(live, key=lambda item: item[1]))

    def _live_screen_transform(self) -> ParallelScreenTransform:
        if self.controller.automatic_updates:
            raise ParallelSectionRigBindingError(
                "the coordinated controller must keep automatic_updates disabled"
            )
        return _camera_screen_transform(
            self.scene.camera,
            display_offset=self.controller.display_offset,
        )

    def apply_section_display_frame(self, frame: SectionDisplayFrame) -> None:
        if not isinstance(frame, SectionDisplayFrame):
            raise TypeError("frame must be a SectionDisplayFrame")
        self._display_frame = frame
        self.controller.update()
        actual = self._actual_painter_order()
        if actual != self._painter_order.draw_order:
            raise ParallelSectionRigBindingError(
                "committed controller painter order differs from preflight evidence"
            )

    def restore_section_display_state(self, snapshot: _DisplaySnapshot) -> None:
        if not isinstance(snapshot, _DisplaySnapshot):
            raise TypeError("snapshot must be a display snapshot")
        self._display_frame = snapshot.frame
        self.controller.restore_transaction_state(snapshot.controller)

    def build_coordinator(self, camera: object) -> ParallelFrameCoordinator:
        """Return the complete audited participant set for ``sequence``."""

        sequence = self.sequence
        if not self.controller.attached:
            raise ParallelSectionRigBindingError(
                "attach the binding before building its playback coordinator"
            )
        if camera is not getattr(self.scene, "camera", None):
            raise ParallelSectionRigBindingError(
                "the playback camera must be this binding's scene.camera"
            )
        if self._coordinator is not None:
            if self._coordinator.poisoned:
                raise ParallelSectionRigBindingError(
                    "the binding's playback coordinator is poisoned"
                )
            return self._coordinator
        gate = parallel_section_preflight_gate(sequence)
        coordinator = ParallelFrameCoordinator()
        coordinator.add(gate.participant())
        coordinator.add(
            parallel_screen_transform_guard(self._live_screen_transform)
        )
        coordinator.add(parallel_camera_frame_participant(camera))
        coordinator.add(section_bank_frame_participant(self))
        if sequence.plane_patch_margin is not None:
            coordinator.add(section_plane_patch_participant(self))
        coordinator.add(section_painter_order_participant(self))
        coordinator.add(section_display_frame_participant(self))
        self._coordinator = coordinator
        return coordinator


def compile_parallel_section_rig_from_shots(
    scene: object,
    timeline: SectionTimeline,
    shot_sequence: ParallelCameraShotSequence,
    initial_camera: ParallelCameraState,
    display_frames: Sequence[SectionDisplayFrame],
    *,
    limits: ParallelPreflightLimits,
    semantic_bank_ids: tuple[str, str],
    start_time: float = 0.0,
    coverage: str = "exact",
    render_times: Sequence[float] | None = None,
    frame_rate: float | None = None,
    plane_patch_margin: float | None = None,
    screen_transforms: Sequence[ParallelScreenTransform] | None = None,
    framing_points_by_frame: Sequence[Sequence[Sequence[float]]] | None = None,
    transition_fraction: float = 0.25,
    transition_mode: SectionTimelineTransitionMode | str = (
        SectionTimelineTransitionMode.CROSSFADE
    ),
    controller_options: Mapping[str, object] | None = None,
) -> ParallelSectionRigBinding:
    """Compile real painter evidence and return one ready-to-attach binding."""

    camera = _require_semantic_scene_camera(scene)
    if _camera_screen_transform(camera) != _IDENTITY_SCREEN_TRANSFORM:
        raise ParallelSectionRigBindingError(
            "the live camera screen transform must be identity before compilation"
        )
    authored_displays = tuple(display_frames)
    authored_transforms = (
        None if screen_transforms is None else tuple(screen_transforms)
    )
    authored_framing = (
        None
        if framing_points_by_frame is None
        else tuple(tuple(tuple(point) for point in frame) for frame in framing_points_by_frame)
    )
    pending = PainterOrderEvidence(
        item_ids=(f"{timeline.section_id}:pending-painter",),
        draw_order=(f"{timeline.section_id}:pending-painter",),
    )
    shared = dict(
        limits=limits,
        semantic_bank_ids=semantic_bank_ids,
        start_time=start_time,
        coverage=coverage,
        render_times=render_times,
        frame_rate=frame_rate,
        plane_patch_margin=plane_patch_margin,
        screen_transforms=authored_transforms,
        framing_points_by_frame=authored_framing,
        transition_fraction=transition_fraction,
        transition_mode=transition_mode,
    )
    draft = compile_parallel_section_sequence_from_shots(
        timeline,
        shot_sequence,
        initial_camera,
        authored_displays,
        painter_orders=lambda _time, _camera, _plane: pending,
        **shared,
    )
    binding = ParallelSectionRigBinding(
        scene,
        draft,
        controller_options=controller_options,
    )
    final = compile_parallel_section_sequence_from_shots(
        timeline,
        shot_sequence,
        initial_camera,
        authored_displays,
        painter_orders=binding.painter_order_provider,
        **shared,
    )
    binding.bind_sequence(final)
    return binding


__all__ = [
    "ParallelSectionRigBinding",
    "ParallelSectionRigBindingError",
    "build_parallel_section_rig_display_catalog",
    "compile_parallel_section_rig_from_shots",
]
