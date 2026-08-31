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
from polyhedron_visibility.quadrics.boundary_compositing import BoundarySourceKind
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.curves import (
    ParametricConicBranch,
    PointMarker3D,
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
from polyhedron_visibility.quadrics.surface_boundaries import (
    GeneratorBoundarySpec,
    SurfaceBoundarySlotDescriptor,
    surface_boundary_slot_descriptors,
)
from polyhedron_visibility.quadrics.semantic_display import (
    SectionDisplayCatalog,
    SectionDisplayFrame,
    SectionDisplayRole,
    SectionSemanticSlot,
)
from polyhedron_visibility.quadrics.semantic_compositing import (
    SectionCompositingFrame,
    SectionDepthPresentationPolicy,
    SectionOcclusionParticipation,
)

from .parallel_camera import ParallelCameraState
from .parallel_frame import (
    ParallelFrameCoordinator,
)
from .parallel_preflight import (
    PainterOrderEvidence,
    ParallelPreflightLimits,
    ParallelScreenTransform,
)
from .parallel_viewport import (
    ParallelViewportState,
    parallel_viewport_frame_participant,
)
from .parallel_shots import ParallelCameraShotSequence
from .quadric_section_parallel import (
    SECTION_PLANE_CHANNEL,
    SECTION_TRANSITION_STATE_CHANNEL,
    ParallelSectionSequence,
    _bank_render_frame,
    _display_catalog_from_frame,
    _timeline_plane_at_time,
    compile_parallel_section_sequence_from_shots,
    parallel_section_preflight_gate,
    section_display_frame_participant,
    section_compositing_frame_participant,
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
_CERTIFIED_BOUNDARY_ROLES = frozenset(
    {
        SectionDisplayRole.GENERATOR,
        SectionDisplayRole.CONTOUR,
        SectionDisplayRole.CAP_RIM,
    }
)
_PAINT_POLICY_BY_PRESENTATION = {
    SectionDepthPresentationPolicy.PHYSICAL: QuadricPaintPolicy.PHYSICAL,
    SectionDepthPresentationPolicy.DIAGRAMMATIC: (
        QuadricPaintPolicy.DIAGRAMMATIC
    ),
    SectionDepthPresentationPolicy.DEPTH_AWARE_DIAGRAMMATIC: (
        QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC
    ),
}


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
    set_zoom = getattr(camera, "set_zoom", None)
    set_frame_center = getattr(camera, "set_parallel_frame_center_xy", None)
    transaction_snapshot = getattr(camera, "snapshot_parallel_transaction", None)
    transaction_restore = getattr(camera, "restore_parallel_transaction", None)
    if not callable(snapshot) or not callable(setter):
        raise ParallelSectionRigBindingError(
            "scene.camera must be a semantic parallel camera with snapshot and "
            "set methods"
        )
    if not callable(get_zoom) or not callable(set_zoom):
        raise ParallelSectionRigBindingError(
            "scene.camera must provide get_zoom() and set_zoom()"
        )
    if not callable(set_frame_center):
        raise ParallelSectionRigBindingError(
            "scene.camera must provide set_parallel_frame_center_xy()"
        )
    if callable(transaction_snapshot) != callable(transaction_restore):
        raise ParallelSectionRigBindingError(
            "camera transaction snapshot and restore methods must be provided together"
        )
    try:
        inherited_zoom = float(get_zoom())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParallelSectionRigBindingError(
            "scene.camera zoom must be finite and positive"
        ) from exc
    if not np.isfinite(inherited_zoom) or inherited_zoom <= 0.0:
        raise ParallelSectionRigBindingError(
            "scene.camera zoom must be finite and positive"
        )
    try:
        frame_center = np.asarray(getattr(camera, "frame_center"), dtype=float)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ParallelSectionRigBindingError(
            "scene.camera frame_center must contain three finite values"
        ) from exc
    if frame_center.shape != (3,) or not np.all(np.isfinite(frame_center)):
        raise ParallelSectionRigBindingError(
            "scene.camera frame_center must contain three finite values"
        )
    camera_token = transaction_snapshot() if callable(transaction_snapshot) else None
    try:
        state = snapshot()
    except Exception as exc:
        raise ParallelSectionRigBindingError(
            f"scene.camera is not in a parallel snapshot state: {exc}"
        ) from exc
    finally:
        if callable(transaction_restore):
            transaction_restore(camera_token)
    if not isinstance(state, ParallelCameraState):
        raise ParallelSectionRigBindingError(
            "scene.camera snapshot did not return ParallelCameraState"
        )
    return camera


def build_parallel_section_rig_display_catalog(
    timeline: SectionTimeline,
    semantic_bank_ids: tuple[str, str],
    *,
    include_plane: bool,
    surface_boundary_mode: str = "certified",
    generator_boundaries: Sequence[GeneratorBoundarySpec] = (),
) -> SectionDisplayCatalog:
    """Build the complete fixed-slot catalog owned by this binding.

    Each topology bank receives two branch slots, one point slot, and
    one cap-chord slot for every analytically possible finite cap chord.
    Certified mode allocates the analytic surface silhouette/rim/generator
    sources instead of the legacy all-purpose surface outline.
    """

    if not isinstance(timeline, SectionTimeline):
        raise TypeError("timeline must be a SectionTimeline")
    if len(semantic_bank_ids) != 2 or len(set(semantic_bank_ids)) != 2:
        raise ParallelSectionRigBindingError(
            "semantic_bank_ids must contain two unique identities"
        )
    prefix = timeline.section_id
    if surface_boundary_mode not in {"certified", "legacy"}:
        raise ParallelSectionRigBindingError(
            "surface_boundary_mode must be 'certified' or 'legacy'"
        )
    slots: list[SectionSemanticSlot] = [
        SectionSemanticSlot(
            f"{prefix}:display:surface-fill",
            SectionDisplayRole.SURFACE_FILL,
        ),
    ]
    if surface_boundary_mode == "legacy":
        slots.append(
            SectionSemanticSlot(
                f"{prefix}:display:surface-outline",
                SectionDisplayRole.SURFACE_OUTLINE,
            )
        )
    else:
        descriptors = surface_boundary_slot_descriptors(
            (timeline.samples[0].surface,),
            generator_boundaries,
        )
        role_by_kind = {
            BoundarySourceKind.SURFACE_SILHOUETTE: SectionDisplayRole.CONTOUR,
            BoundarySourceKind.SURFACE_CAP_RIM: SectionDisplayRole.CAP_RIM,
            BoundarySourceKind.SURFACE_TRIM_RIM: SectionDisplayRole.CAP_RIM,
            BoundarySourceKind.SURFACE_GENERATOR: SectionDisplayRole.GENERATOR,
        }
        slots.extend(
            SectionSemanticSlot(
                descriptor.source_id,
                role_by_kind[descriptor.source_kind],
                source_id=descriptor.source_id,
            )
            for descriptor in descriptors
        )
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
    points: tuple[PointMarker3D, ...]
    base_opacities: tuple[tuple[str, float], ...]
    base_point_opacities: tuple[tuple[str, float], ...]
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
        generator_boundaries: Sequence[GeneratorBoundarySpec] = (),
    ) -> None:
        if not isinstance(draft_sequence, ParallelSectionSequence):
            raise TypeError("draft_sequence must be a ParallelSectionSequence")
        self.scene = scene
        camera = _require_semantic_scene_camera(scene)
        self._draft_sequence = draft_sequence
        self._generator_boundaries = tuple(generator_boundaries)
        if not all(
            isinstance(item, GeneratorBoundarySpec)
            for item in self._generator_boundaries
        ):
            raise TypeError(
                "generator_boundaries must contain GeneratorBoundarySpec values"
            )
        surface = draft_sequence.timeline.samples[0].surface
        expected_descriptors = surface_boundary_slot_descriptors(
            (surface,),
            self._generator_boundaries,
        )
        authored_boundary_slots = tuple(
            item
            for item in draft_sequence.display_frames[0].slots
            if item.role in _CERTIFIED_BOUNDARY_ROLES
        )
        self._certified_surface_boundaries = bool(authored_boundary_slots)
        if self._generator_boundaries and not self._certified_surface_boundaries:
            raise ParallelSectionRigBindingError(
                "generator_boundaries require a certified surface-boundary catalog"
            )
        if self._certified_surface_boundaries:
            authored_by_source = {
                item.source_id: item for item in authored_boundary_slots
            }
            expected_by_source = {
                item.source_id: item for item in expected_descriptors
            }
            if set(authored_by_source) != set(expected_by_source):
                raise ParallelSectionRigBindingError(
                    "certified surface-boundary catalog differs from the "
                    "surface/generator allocation"
                )
            self._boundary_display_slot_by_source = {
                source_id: item.slot_id
                for source_id, item in authored_by_source.items()
            }
        else:
            self._boundary_display_slot_by_source = {}
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
        self._point_slots_by_bank = {
            bank_id: tuple(
                sorted(
                    item.slot_id
                    for item in first_display.slots
                    if item.role is SectionDisplayRole.SECTION_POINT
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
        self._allocated_point_ids = tuple(
            sorted(
                point_id
                for values in self._point_slots_by_bank.values()
                for point_id in values
            )
        )

        self._bank_frame = draft_sequence.bank_render_frames[0]
        self._display_frame = draft_sequence.display_frames[0]
        self._compositing_frame = draft_sequence.compositing_frames[0]
        self._patch_fit = draft_sequence.plane_patch_fits[0]
        self._painter_order = draft_sequence.painter_orders[0]
        self._curves: tuple[AnalyticCurve3D, ...] = ()
        self._points: tuple[PointMarker3D, ...] = ()
        self._base_curve_opacities: dict[str, float] = {}
        self._base_point_opacities: dict[str, float] = {}
        self._projection_override: ParallelCameraState | ParallelViewportState | None = (
            ParallelViewportState(
                draft_sequence.camera_samples[0].state,
                draft_sequence.screen_transforms[0],
            )
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
        options["style"] = style
        owned = {
            "surfaces",
            "curves",
            "points",
            "projection",
            "allocated_curve_ids",
            "curve_opacities",
            "allocated_point_ids",
            "point_opacities",
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
            "boundary_opacities",
            "generator_boundaries",
            "allocated_boundary_ids",
            "surface_opacities",
            "surface_stroke_opacities",
            "occluding_surface_ids",
            "paint_policy",
            "section_plane_fill_opacity",
            "section_plane_stroke_opacity",
        }
        conflicts = tuple(sorted(owned & set(options)))
        if conflicts:
            raise ParallelSectionRigBindingError(
                "controller_options cannot override binding-owned keys: "
                + ", ".join(conflicts)
            )
        section_enabled = self._patch_fit is not None
        self.controller = QuadricOcclusion3D(
            scene,
            surfaces=(surface,),
            surface_opacities=self._resolve_surface_opacities,
            surface_stroke_opacities=(
                self._resolve_surface_stroke_opacities
            ),
            occluding_surface_ids=self._resolve_occluding_surface_ids,
            curves=self._resolve_curves,
            points=self._resolve_points,
            projection=self._resolve_projection,
            allocated_curve_ids=self._allocated_curve_ids,
            curve_opacities=self._resolve_curve_opacities,
            allocated_point_ids=self._allocated_point_ids,
            point_opacities=self._resolve_point_opacities,
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
            include_surface_boundaries=self._certified_surface_boundaries,
            generator_boundaries=self._generator_boundaries,
            boundary_opacities=self._resolve_boundary_opacities,
            paint_policy=self._resolve_paint_policy,
            section_plane_fill_opacity=(
                self._resolve_section_plane_fill_opacity
            ),
            section_plane_stroke_opacity=(
                self._resolve_section_plane_stroke_opacity
            ),
            display_offset=(0.0, 0.0),
            automatic_updates=False,
            legacy_surface_stroke_fallback=(
                not self._certified_surface_boundaries
            ),
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
    def allocated_point_ids(self) -> tuple[str, ...]:
        return self._allocated_point_ids

    @property
    def attached(self) -> bool:
        return self.controller.attached

    @property
    def display_mobject(self):
        return self.controller.display_mobject

    def _validate_compositing_frame_contract(
        self,
        frame: SectionCompositingFrame,
        display: SectionDisplayFrame,
    ) -> None:
        if not isinstance(frame, SectionCompositingFrame):
            raise TypeError("frame must be a SectionCompositingFrame")
        if not isinstance(display, SectionDisplayFrame):
            raise TypeError("display must be a SectionDisplayFrame")
        try:
            frame.validate_catalog(_display_catalog_from_frame(display))
        except (TypeError, ValueError) as exc:
            raise ParallelSectionRigBindingError(
                f"compositing frame does not match the fixed display catalog: {exc}"
            ) from exc
        policies = {item.depth_presentation for item in frame.slots}
        if len(policies) != 1:
            raise ParallelSectionRigBindingError(
                "one Rig frame must use one depth presentation policy"
            )
        unsupported_paint_only = tuple(
            item.role.value
            for item in frame.slots
            if (
                item.occlusion_participation
                is SectionOcclusionParticipation.PAINT_ONLY
                and item.role is not SectionDisplayRole.SURFACE_FILL
            )
        )
        if unsupported_paint_only:
            raise ParallelSectionRigBindingError(
                "paint-only occlusion participation is currently supported "
                "only by the surface-fill slot; invalid roles="
                + ", ".join(sorted(set(unsupported_paint_only)))
            )

    def _validate_sequence_contract(self, sequence: ParallelSectionSequence) -> None:
        first = sequence.display_frames[0]
        first_metadata = tuple(
            (item.slot_id, item.role, item.source_id, item.topology_bank)
            for item in first.slots
        )
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
        for display, frame in zip(
            sequence.display_frames,
            sequence.compositing_frames,
        ):
            self._validate_compositing_frame_contract(frame, display)
        required_roles = {
            SectionDisplayRole.SURFACE_FILL,
        }
        if not self._certified_surface_boundaries:
            required_roles.add(SectionDisplayRole.SURFACE_OUTLINE)
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
            point_slots = tuple(
                item
                for item in first.slots
                if item.role is SectionDisplayRole.SECTION_POINT
                and item.topology_bank == bank_id
            )
            required_points = max(
                (
                    layer.isolated_point_count
                    for frame in sequence.bank_render_frames
                    for layer in frame.layers
                    if layer.semantic_bank_id == bank_id
                ),
                default=0,
            )
            if len(point_slots) < required_points:
                raise ParallelSectionRigBindingError(
                    f"topology bank {bank_id!r} requires {required_points} "
                    "fixed section-point slot(s)"
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
            and left.compositing_frames == right.compositing_frames
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

    def _resolve_points(self) -> tuple[PointMarker3D, ...]:
        return self._points

    def _effective_slot_opacities(self) -> dict[str, float]:
        display = {
            item.slot_id: item.opacity_multiplier
            for item in self._display_frame.slots
        }
        compositing = {
            item.slot_id: item.display_opacity
            for item in self._compositing_frame.slots
        }
        if set(display) != set(compositing):
            raise ParallelSectionRigBindingError(
                "display and compositing frames use different fixed slots"
            )
        return {
            slot_id: display[slot_id] * compositing[slot_id]
            for slot_id in display
        }

    def _compositing_state_for_role(self, role: SectionDisplayRole):
        states = tuple(
            item for item in self._compositing_frame.slots
            if item.role is role
        )
        if len(states) != 1:
            raise ParallelSectionRigBindingError(
                f"the compositing frame requires exactly one {role.value!r} slot"
            )
        return states[0]

    def _resolve_surface_opacities(self) -> Mapping[str, float]:
        surface_id = self._draft_sequence.timeline.surface_id
        slot = self._compositing_state_for_role(
            SectionDisplayRole.SURFACE_FILL
        )
        return {surface_id: self._effective_slot_opacities()[slot.slot_id]}

    def _resolve_surface_stroke_opacities(self) -> Mapping[str, float]:
        surface_id = self._draft_sequence.timeline.surface_id
        states = tuple(
            item
            for item in self._compositing_frame.slots
            if item.role is SectionDisplayRole.SURFACE_OUTLINE
        )
        if not states:
            return {surface_id: 1.0}
        if len(states) != 1:
            raise ParallelSectionRigBindingError(
                "the compositing frame has duplicate surface-outline slots"
            )
        return {
            surface_id: self._effective_slot_opacities()[states[0].slot_id]
        }

    def _resolve_occluding_surface_ids(self) -> tuple[str, ...]:
        state = self._compositing_state_for_role(
            SectionDisplayRole.SURFACE_FILL
        )
        if (
            state.occlusion_participation
            is SectionOcclusionParticipation.CERTIFIED
        ):
            return (self._draft_sequence.timeline.surface_id,)
        return ()

    def _resolve_paint_policy(self) -> QuadricPaintPolicy:
        policies = {
            item.depth_presentation for item in self._compositing_frame.slots
        }
        if len(policies) != 1:
            raise ParallelSectionRigBindingError(
                "one Rig frame must use one depth presentation policy"
            )
        return _PAINT_POLICY_BY_PRESENTATION[next(iter(policies))]

    def _resolve_section_plane_fill_opacity(self) -> float:
        states = tuple(
            item for item in self._compositing_frame.slots
            if item.role is SectionDisplayRole.PLANE_FILL
        )
        if not states:
            return 0.0
        if len(states) != 1:
            raise ParallelSectionRigBindingError(
                "the compositing frame has duplicate plane-fill slots"
            )
        return self._effective_slot_opacities()[states[0].slot_id]

    def _resolve_section_plane_stroke_opacity(self) -> float:
        states = tuple(
            item for item in self._compositing_frame.slots
            if item.role is SectionDisplayRole.PLANE_OUTLINE
        )
        if not states:
            return 0.0
        if len(states) != 1:
            raise ParallelSectionRigBindingError(
                "the compositing frame has duplicate plane-outline slots"
            )
        return self._effective_slot_opacities()[states[0].slot_id]

    def _resolve_curve_opacities(self) -> Mapping[str, float]:
        display_opacity = self._effective_slot_opacities()
        return {
            physical_id: base_opacity
            * display_opacity[self._physical_to_semantic[physical_id]]
            for physical_id, base_opacity in self._base_curve_opacities.items()
        }

    def _resolve_point_opacities(self) -> Mapping[str, float]:
        display_opacity = self._effective_slot_opacities()
        return {
            point_id: base_opacity * display_opacity[point_id]
            for point_id, base_opacity in self._base_point_opacities.items()
        }

    def _resolve_boundary_opacities(self) -> Mapping[str, float]:
        display_opacity = self._effective_slot_opacities()
        plane_outline = tuple(
            item.slot_id for item in self._display_frame.slots
            if item.role is SectionDisplayRole.PLANE_OUTLINE
        )
        result: dict[str, float] = {}
        for source_id in self.controller.allocated_boundary_ids:
            slot_id = self._boundary_display_slot_by_source.get(source_id)
            result[source_id] = (
                (
                    display_opacity[plane_outline[0]]
                    if slot_id is None
                    and source_id.startswith("boundary:plane:")
                    and len(plane_outline) == 1
                    else 1.0
                )
                if slot_id is None
                else display_opacity[slot_id]
            )
        return result

    def _resolve_projection(
        self,
        _scene: object,
    ) -> ParallelCameraState | ParallelViewportState:
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
        points: list[PointMarker3D] = []
        base_opacities: dict[str, float] = {}
        base_point_opacities: dict[str, float] = {}
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
            isolated = boundary.trace.isolated_world_points
            if len(isolated) != layer.isolated_point_count:
                raise ParallelSectionRigBindingError(
                    "materialized isolated-point count differs from bank evidence"
                )
            point_slots = self._point_slots_by_bank[layer.semantic_bank_id]
            if len(isolated) > len(point_slots):
                raise ParallelSectionRigBindingError(
                    "isolated section points exceed the fixed bank capacity"
                )
            for point_index, point in enumerate(isolated):
                point_id = point_slots[point_index]
                points.append(PointMarker3D(point_id, point))
                base_point_opacities[point_id] = layer.opacity
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
        point_identities = tuple(item.point_id for item in points)
        if len(set(point_identities)) != len(point_identities):
            raise ParallelSectionRigBindingError(
                "active topology banks produced duplicate physical point ids"
            )
        self._curves = tuple(sorted(curves, key=lambda item: item.curve_id))
        self._points = tuple(sorted(points, key=lambda item: item.point_id))
        self._base_curve_opacities = base_opacities
        self._base_point_opacities = base_point_opacities
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
        self._projection_override = ParallelViewportState(
            camera,
            self._draft_sequence.screen_transforms[index],
        )
        self.controller.display_offset = (0.0, 0.0)
        self._bank_frame = self._draft_sequence.bank_render_frames[index]
        self._display_frame = self._draft_sequence.display_frames[index]
        self._compositing_frame = self._draft_sequence.compositing_frames[index]
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
        self._projection_override = ParallelViewportState(
            sequence.camera_samples[0].state,
            sequence.screen_transforms[0],
        )
        self.controller.display_offset = (0.0, 0.0)
        self._bank_frame = sequence.bank_render_frames[0]
        self._display_frame = sequence.display_frames[0]
        self._compositing_frame = sequence.compositing_frames[0]
        self._patch_fit = sequence.plane_patch_fits[0]
        self._painter_order = sequence.painter_orders[0]
        self._apply_bank_geometry(self._bank_frame)

    def attach(self) -> "ParallelSectionRigBinding":
        sequence = self.sequence
        if self.controller.attached:
            return self
        # Compilation certifies the camera state which existed at that time,
        # but callers may keep authoring the Scene before ownership begins.
        # Revalidate the complete live viewport contract immediately before
        # the controller can add any Mobject to the Scene.
        _require_semantic_scene_camera(self.scene)
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
            self._points,
            tuple(sorted(self._base_curve_opacities.items())),
            tuple(sorted(self._base_point_opacities.items())),
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
        self._points = snapshot.points
        self._base_curve_opacities = dict(snapshot.base_opacities)
        self._base_point_opacities = dict(snapshot.base_point_opacities)
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

    def snapshot_section_compositing_state(self) -> SectionCompositingFrame:
        return self._compositing_frame

    def apply_section_compositing_frame(
        self,
        frame: SectionCompositingFrame,
    ) -> None:
        if not isinstance(frame, SectionCompositingFrame):
            raise TypeError("frame must be a SectionCompositingFrame")
        self._validate_compositing_frame_contract(frame, self._display_frame)
        self._compositing_frame = frame

    def restore_section_compositing_state(
        self,
        frame: SectionCompositingFrame,
    ) -> None:
        if not isinstance(frame, SectionCompositingFrame):
            raise TypeError("snapshot must be a SectionCompositingFrame")
        self._compositing_frame = frame

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
            parallel_viewport_frame_participant(
                camera,
                display_offset_getter=lambda: self.controller.display_offset,
                display_offset_setter=lambda value: setattr(
                    self.controller,
                    "display_offset",
                    value,
                ),
            )
        )
        coordinator.add(section_bank_frame_participant(self))
        if sequence.plane_patch_margin is not None:
            coordinator.add(section_plane_patch_participant(self))
        coordinator.add(section_painter_order_participant(self))
        coordinator.add(section_compositing_frame_participant(self))
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
    generator_boundaries: Sequence[GeneratorBoundarySpec] = (),
    compositing_frames: Sequence[SectionCompositingFrame] | None = None,
) -> ParallelSectionRigBinding:
    """Compile real painter evidence and return one ready-to-attach binding."""

    _require_semantic_scene_camera(scene)
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
        compositing_frames=(
            None if compositing_frames is None else tuple(compositing_frames)
        ),
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
        generator_boundaries=generator_boundaries,
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
