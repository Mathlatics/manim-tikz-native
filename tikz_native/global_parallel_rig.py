"""One global quadric compositor for several parallel section Rigs.

The individual :class:`ParallelSectionRigBinding` objects remain unattached and
act only as source-authoritative geometry providers.  This module allocates one
new :class:`QuadricOcclusion3D` over their union, so curves, point markers, and
certified surface boundaries from different Rigs participate in the same
visibility solve and the same managed painter band.

Version one deliberately excludes finite cutting-plane fills/outlines.  The
quadric controller's section compositor supports one section plane, whereas a
multi-Rig frame may contain several independent planes; accepting those as
ordinary paint objects would silently reduce global ordering to block sorting.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from polyhedron_visibility.quadrics.manim import QuadricOcclusion3D
from polyhedron_visibility.quadrics.semantic_display import SectionDisplayRole

from .parallel_camera import ParallelCameraState
from .parallel_frame import (
    ParallelFrameBindingKind,
    ParallelFrameCoordinator,
    ParallelFrameParticipant,
    ParallelFramePhase,
    ParallelFrameState,
)
from .parallel_preflight import PainterOrderEvidence, ParallelScreenTransform
from .parallel_viewport import (
    PARALLEL_VIEWPORT_TRANSFORM_CHANNEL,
    ParallelViewportState,
    parallel_viewport_frame_participant,
)
from .quadric_section_parallel_rig import (
    ParallelSectionRigBinding,
    ParallelSectionRigBindingError,
    _require_semantic_scene_camera,
)


GLOBAL_PARALLEL_RIG_STAGE_CHANNEL = "global-parallel-rig-stage"
GLOBAL_PARALLEL_RIG_SCHEMA = "tikz-native-global-parallel-rig/v1"


class GlobalParallelRigError(RuntimeError):
    """Several local Rigs cannot be certified as one global transaction."""


def _digest_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


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
        raise GlobalParallelRigError(
            "the global quadric controller produced an empty painter order"
        )
    return PainterOrderEvidence(
        item_ids=order,
        relations=tuple(zip(order, order[1:])),
        draw_order=order,
    )


@dataclass(frozen=True, slots=True)
class GlobalParallelRigFrameEvidence:
    """Renderer-neutral proof tying all local states to one global painter plan."""

    frame_index: int
    time: float
    rig_state_digests: tuple[tuple[str, str], ...]
    painter_order: PainterOrderEvidence
    digest: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
            raise TypeError("frame_index must be an integer")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        time = float(self.time)
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        object.__setattr__(self, "time", time)
        values = tuple(self.rig_state_digests)
        if not values or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(part, str) and part for part in item)
            for item in values
        ):
            raise TypeError("rig_state_digests must contain identity/digest pairs")
        if len({item[0] for item in values}) != len(values):
            raise ValueError("rig_state_digests contain duplicate Rig identities")
        object.__setattr__(self, "rig_state_digests", values)
        if not isinstance(self.painter_order, PainterOrderEvidence):
            raise TypeError("painter_order must be PainterOrderEvidence")
        expected = _digest_json(self._payload())
        if self.digest and self.digest != expected:
            raise ValueError("global frame evidence digest is inconsistent")
        object.__setattr__(self, "digest", expected)

    def _payload(self) -> dict[str, object]:
        return {
            "frameIndex": self.frame_index,
            "time": self.time,
            "rigStates": [list(item) for item in self.rig_state_digests],
            "painterOrder": self.painter_order.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "digest": self.digest}


@dataclass(frozen=True, slots=True)
class GlobalParallelRigStageFrame:
    """All immutable local provider values required by one global frame."""

    bank_frames: tuple[object, ...]
    display_frames: tuple[object, ...]
    compositing_frames: tuple[object, ...]
    local_painter_orders: tuple[PainterOrderEvidence, ...]
    evidence: GlobalParallelRigFrameEvidence

    def __post_init__(self) -> None:
        for name in (
            "bank_frames",
            "display_frames",
            "compositing_frames",
            "local_painter_orders",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        count = len(self.bank_frames)
        if count < 2 or any(
            len(getattr(self, name)) != count
            for name in (
                "display_frames",
                "compositing_frames",
                "local_painter_orders",
            )
        ):
            raise GlobalParallelRigError(
                "global stage frames must cover the same two or more Rigs"
            )
        if not isinstance(self.evidence, GlobalParallelRigFrameEvidence):
            raise TypeError("evidence must be GlobalParallelRigFrameEvidence")


@dataclass(frozen=True, slots=True)
class GlobalParallelRigSequence:
    """Fixed evaluation grid plus global painter evidence for playback."""

    evaluation_times: tuple[float, ...]
    rig_ids: tuple[str, ...]
    evidences: tuple[GlobalParallelRigFrameEvidence, ...]
    frames: tuple[ParallelFrameState, ...]
    digest: str = ""
    schema: str = GLOBAL_PARALLEL_RIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GLOBAL_PARALLEL_RIG_SCHEMA:
            raise GlobalParallelRigError("invalid global parallel Rig schema")
        for name in ("evaluation_times", "rig_ids", "evidences", "frames"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        count = len(self.evaluation_times)
        if not count or any(
            right <= left
            for left, right in zip(self.evaluation_times, self.evaluation_times[1:])
        ):
            raise GlobalParallelRigError(
                "evaluation_times must be non-empty and strictly increasing"
            )
        if len(self.rig_ids) < 2 or len(set(self.rig_ids)) != len(self.rig_ids):
            raise GlobalParallelRigError("rig_ids must contain two or more unique ids")
        if len(self.evidences) != count or len(self.frames) != count:
            raise GlobalParallelRigError(
                "global evidence and frames must cover the evaluation grid"
            )
        for index, (time, evidence, frame) in enumerate(
            zip(self.evaluation_times, self.evidences, self.frames)
        ):
            if (
                evidence.frame_index != index
                or evidence.time != time
                or tuple(item[0] for item in evidence.rig_state_digests)
                != self.rig_ids
                or frame.preflight_input_digest != evidence.digest
            ):
                raise GlobalParallelRigError(
                    "global frame/evidence identity differs from its evaluation slot"
                )
            stage = frame.channel(GLOBAL_PARALLEL_RIG_STAGE_CHANNEL)
            if not isinstance(stage, GlobalParallelRigStageFrame):
                raise GlobalParallelRigError("global frame lacks its stage payload")
            if stage.evidence != evidence:
                raise GlobalParallelRigError(
                    "global stage payload differs from frame evidence"
                )
        expected = _digest_json(
            {
                "schema": self.schema,
                "evaluationTimes": list(self.evaluation_times),
                "rigIds": list(self.rig_ids),
                "frames": [item.to_dict() for item in self.evidences],
            }
        )
        if self.digest and self.digest != expected:
            raise GlobalParallelRigError("global sequence digest is inconsistent")
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class _LocalStageSnapshot:
    bank: object
    display: object
    compositing: object
    painter: object
    projection_override: object


class GlobalParallelRigBinding:
    """One global controller over several compiled, unattached local Rigs."""

    def __init__(self, bindings: Sequence[ParallelSectionRigBinding]) -> None:
        values = tuple(bindings)
        if len(values) < 2 or not all(
            isinstance(item, ParallelSectionRigBinding) for item in values
        ):
            raise TypeError(
                "bindings must contain two or more ParallelSectionRigBinding values"
            )
        self.bindings = values
        self.scene = values[0].scene
        self._projection_override: ParallelViewportState | None = None
        self._coordinator: ParallelFrameCoordinator[ParallelFrameState] | None = None
        self._validate_contract()

        first_controller = values[0].controller
        surfaces = tuple(
            item.sequence.timeline.samples[0].surface for item in values
        )
        generators = tuple(
            generator
            for item in values
            for generator in item._generator_boundaries
        )
        constraints = tuple(
            constraint
            for item in values
            for constraint in item.controller.surface_constraints
        )
        requested_band = first_controller._band._requested_band
        self.controller = QuadricOcclusion3D(
            self.scene,
            surfaces=surfaces,
            surface_opacities=self._resolve_surface_opacities,
            surface_stroke_opacities=self._resolve_surface_stroke_opacities,
            occluding_surface_ids=self._resolve_occluding_surface_ids,
            curves=self._resolve_curves,
            points=self._resolve_points,
            projection=self._resolve_projection,
            paint_policy=self._resolve_paint_policy,
            style=first_controller.style,
            boundary_styles=first_controller.boundary_styles,
            boundary_opacities=self._resolve_boundary_opacities,
            limits=first_controller.limits,
            max_chord_error=first_controller.max_chord_error,
            context=first_controller.context,
            painter_z_band=requested_band,
            surface_constraints=constraints,
            surface_order_mode="automatic",
            allocated_curve_ids=tuple(
                sorted(
                    curve_id
                    for item in values
                    for curve_id in item.allocated_curve_ids
                )
            ),
            curve_opacities=self._resolve_curve_opacities,
            allocated_point_ids=tuple(
                sorted(
                    point_id
                    for item in values
                    for point_id in item.allocated_point_ids
                )
            ),
            point_opacities=self._resolve_point_opacities,
            section_id=None,
            section_plane=None,
            section_patch=None,
            section_compositing_limits=first_controller.section_compositing_limits,
            boundary_section_limits=first_controller.boundary_section_limits,
            boundary_visibility_mode="unified",
            include_surface_boundaries=True,
            generator_boundaries=generators,
            display_offset=(0.0, 0.0),
            automatic_updates=False,
            legacy_surface_stroke_fallback=False,
        )
        self.sequence = self._compile_global_sequence()
        self._stage_index(0, projection_override=True)

    @property
    def attached(self) -> bool:
        return self.controller.attached

    @property
    def display_mobject(self):
        return self.controller.display_mobject

    def _validate_contract(self) -> None:
        first = self.bindings[0]
        try:
            first_sequence = first.sequence
        except Exception as exc:
            raise GlobalParallelRigError(
                "every local Rig must be compiled and bound before aggregation"
            ) from exc
        first_controller = first.controller
        seen_surfaces: set[str] = set()
        seen_curves: set[str] = set()
        seen_points: set[str] = set()
        for binding in self.bindings:
            try:
                sequence = binding.sequence
            except Exception as exc:
                raise GlobalParallelRigError(
                    "every local Rig must be compiled and bound before aggregation"
                ) from exc
            if binding.scene is not self.scene:
                raise GlobalParallelRigError("all local Rigs must share one Scene")
            if binding.controller.attached:
                raise GlobalParallelRigError(
                    "local Rig controllers must remain unattached"
                )
            if not binding._certified_surface_boundaries:
                raise GlobalParallelRigError(
                    "global aggregation requires certified surface boundaries"
                )
            if sequence.evaluation_times != first_sequence.evaluation_times:
                raise GlobalParallelRigError(
                    "all local Rigs must share one evaluation grid"
                )
            if sequence.screen_transforms != first_sequence.screen_transforms:
                raise GlobalParallelRigError(
                    "all local Rigs must share one screen transform per frame"
                )
            if any(
                not _camera_equal(left.state, right.state)
                for left, right in zip(
                    sequence.camera_samples,
                    first_sequence.camera_samples,
                )
            ):
                raise GlobalParallelRigError(
                    "all local Rigs must share one camera state per frame"
                )
            if sequence.plane_patch_margin is not None or any(
                item is not None for item in sequence.plane_patch_fits
            ):
                raise GlobalParallelRigError(
                    "v1 global aggregation does not support finite plane patches"
                )
            unsupported_roles = {
                slot.role
                for frame in sequence.display_frames
                for slot in frame.slots
                if slot.role
                in (SectionDisplayRole.PLANE_FILL, SectionDisplayRole.PLANE_OUTLINE)
            }
            if unsupported_roles:
                raise GlobalParallelRigError(
                    "v1 global aggregation rejects plane-fill and plane-outline roles"
                )
            if (
                binding.controller.style != first_controller.style
                or binding.controller.limits != first_controller.limits
                or binding.controller.boundary_styles
                != first_controller.boundary_styles
                or binding.controller.max_chord_error
                != first_controller.max_chord_error
                or binding.controller.context != first_controller.context
                or binding.controller.section_compositing_limits
                != first_controller.section_compositing_limits
                or binding.controller.boundary_section_limits
                != first_controller.boundary_section_limits
                or binding.controller._band._requested_band
                != first_controller._band._requested_band
            ):
                raise GlobalParallelRigError(
                    "all local Rigs must share global controller style/limit policy"
                )
            surface_id = sequence.timeline.samples[0].surface.surface_id
            if surface_id in seen_surfaces:
                raise GlobalParallelRigError("surface identities collide across Rigs")
            seen_surfaces.add(surface_id)
            collision = seen_curves.intersection(binding.allocated_curve_ids)
            if collision:
                raise GlobalParallelRigError(
                    "curve identities collide across Rigs: "
                    + ", ".join(sorted(collision))
                )
            seen_curves.update(binding.allocated_curve_ids)
            point_collision = seen_points.intersection(binding.allocated_point_ids)
            if point_collision:
                raise GlobalParallelRigError(
                    "point identities collide across Rigs: "
                    + ", ".join(sorted(point_collision))
                )
            seen_points.update(binding.allocated_point_ids)

    def _rig_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{item.sequence.timeline.section_id}@"
            f"{item.sequence.timeline.samples[0].surface.surface_id}"
            for item in self.bindings
        )

    def _rig_state_digest(self, binding: ParallelSectionRigBinding, index: int) -> str:
        sequence = binding.sequence
        source = sequence.frames[index]
        return _digest_json(
            {
                "sourceFrameId": source.frame_id,
                "sourcePreflightDigest": source.preflight_input_digest,
                "bank": sequence.bank_render_frames[index].digest,
                "display": sequence.display_frames[index].digest,
                "compositing": sequence.compositing_frames[index].digest,
                "screenTransform": sequence.screen_transforms[index].to_dict(),
            }
        )

    def _capture_local_stage(self) -> tuple[_LocalStageSnapshot, ...]:
        return tuple(
            _LocalStageSnapshot(
                item.snapshot_section_bank_render_state(),
                item._display_frame,
                item.snapshot_section_compositing_state(),
                item.snapshot_section_painter_order_state(),
                item._projection_override,
            )
            for item in self.bindings
        )

    def _restore_local_stage(
        self,
        snapshots: Sequence[_LocalStageSnapshot],
    ) -> None:
        if len(snapshots) != len(self.bindings):
            raise GlobalParallelRigError("local stage snapshot count changed")
        for binding, snapshot in zip(self.bindings, snapshots):
            binding.restore_section_bank_render_state(snapshot.bank)
            binding._display_frame = snapshot.display
            binding.restore_section_compositing_state(snapshot.compositing)
            binding.restore_section_painter_order_state(snapshot.painter)
            binding._projection_override = snapshot.projection_override

    def _stage_payload(
        self,
        payload: GlobalParallelRigStageFrame,
        *,
        projection_override: bool,
    ) -> None:
        if len(payload.bank_frames) != len(self.bindings):
            raise GlobalParallelRigError("global stage Rig count changed")
        for index, binding in enumerate(self.bindings):
            binding.apply_section_bank_render_frame(payload.bank_frames[index])
            binding._display_frame = payload.display_frames[index]
            binding.apply_section_compositing_frame(
                payload.compositing_frames[index]
            )
            binding.apply_section_painter_order(payload.local_painter_orders[index])
        if projection_override:
            sequence = self.bindings[0].sequence
            frame_index = payload.evidence.frame_index
            self._projection_override = ParallelViewportState(
                sequence.camera_samples[frame_index].state,
                sequence.screen_transforms[frame_index],
            )
        else:
            self._projection_override = None

    def _stage_index(self, index: int, *, projection_override: bool) -> None:
        self._stage_payload(
            self.sequence.frames[index].channel(GLOBAL_PARALLEL_RIG_STAGE_CHANNEL),
            projection_override=projection_override,
        )

    def _compile_global_sequence(self) -> GlobalParallelRigSequence:
        rig_ids = self._rig_ids()
        first = self.bindings[0].sequence
        original = self._capture_local_stage()
        previous_override = self._projection_override
        evidences: list[GlobalParallelRigFrameEvidence] = []
        payload_parts: list[tuple[tuple[object, ...], ...]] = []
        try:
            for index, time in enumerate(first.evaluation_times):
                bank_frames = tuple(
                    item.sequence.bank_render_frames[index] for item in self.bindings
                )
                display_frames = tuple(
                    item.sequence.display_frames[index] for item in self.bindings
                )
                compositing_frames = tuple(
                    item.sequence.compositing_frames[index] for item in self.bindings
                )
                local_orders = tuple(
                    item.sequence.painter_orders[index] for item in self.bindings
                )
                temporary = GlobalParallelRigFrameEvidence(
                    index,
                    time,
                    tuple(
                        (rig_id, self._rig_state_digest(binding, index))
                        for rig_id, binding in zip(rig_ids, self.bindings)
                    ),
                    PainterOrderEvidence(),
                )
                stage = GlobalParallelRigStageFrame(
                    bank_frames,
                    display_frames,
                    compositing_frames,
                    local_orders,
                    temporary,
                )
                self._stage_payload(stage, projection_override=True)
                numeric = self.controller._prepare_numeric(None)
                evidence = GlobalParallelRigFrameEvidence(
                    index,
                    time,
                    temporary.rig_state_digests,
                    _painter_evidence(numeric.painter_draw_order),
                )
                evidences.append(evidence)
                payload_parts.append(
                    (bank_frames, display_frames, compositing_frames, local_orders)
                )
        finally:
            self._restore_local_stage(original)
            self._projection_override = previous_override

        frames: list[ParallelFrameState] = []
        for index, evidence in enumerate(evidences):
            bank_frames, display_frames, compositing_frames, local_orders = (
                payload_parts[index]
            )
            stage = GlobalParallelRigStageFrame(
                bank_frames,
                display_frames,
                compositing_frames,
                local_orders,
                evidence,
            )
            frames.append(
                ParallelFrameState(
                    first.camera_samples[index].state,
                    MappingProxyType(
                        {
                            PARALLEL_VIEWPORT_TRANSFORM_CHANNEL: (
                                first.screen_transforms[index]
                            ),
                            GLOBAL_PARALLEL_RIG_STAGE_CHANNEL: stage,
                        }
                    ),
                    frame_id=f"global-parallel-rig:{index}:{evidence.digest[7:19]}",
                    preflight_input_digest=evidence.digest,
                )
            )
        return GlobalParallelRigSequence(
            first.evaluation_times,
            rig_ids,
            tuple(evidences),
            tuple(frames),
        )

    def _resolve_projection(self, _scene: object) -> ParallelViewportState | ParallelCameraState:
        if self._projection_override is not None:
            return self._projection_override
        snapshot = getattr(self.scene.camera, "snapshot_parallel_state", None)
        if not callable(snapshot):
            raise GlobalParallelRigError(
                "scene.camera must provide snapshot_parallel_state()"
            )
        value = snapshot()
        if not isinstance(value, ParallelCameraState):
            raise GlobalParallelRigError(
                "scene.camera returned an invalid parallel camera state"
            )
        return value

    def _resolve_curves(self):
        return tuple(
            curve for binding in self.bindings for curve in binding._resolve_curves()
        )

    def _resolve_points(self):
        return tuple(
            point for binding in self.bindings for point in binding._resolve_points()
        )

    def _merge_mappings(self, method_name: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for binding in self.bindings:
            values = getattr(binding, method_name)()
            overlap = set(result).intersection(values)
            if overlap:
                raise GlobalParallelRigError(
                    f"global opacity identities collide: {', '.join(sorted(overlap))}"
                )
            result.update(values)
        return result

    def _resolve_surface_opacities(self) -> Mapping[str, float]:
        return self._merge_mappings("_resolve_surface_opacities")

    def _resolve_surface_stroke_opacities(self) -> Mapping[str, float]:
        return self._merge_mappings("_resolve_surface_stroke_opacities")

    def _resolve_curve_opacities(self) -> Mapping[str, float]:
        return self._merge_mappings("_resolve_curve_opacities")

    def _resolve_point_opacities(self) -> Mapping[str, float]:
        return self._merge_mappings("_resolve_point_opacities")

    def _resolve_boundary_opacities(self) -> Mapping[str, float]:
        return self._merge_mappings("_resolve_boundary_opacities")

    def _resolve_occluding_surface_ids(self) -> tuple[str, ...]:
        return tuple(
            surface_id
            for binding in self.bindings
            for surface_id in binding._resolve_occluding_surface_ids()
        )

    def _resolve_paint_policy(self):
        values = tuple(binding._resolve_paint_policy() for binding in self.bindings)
        if len(set(values)) != 1:
            raise GlobalParallelRigError(
                "all local Rigs must use one depth presentation policy per frame"
            )
        return values[0]

    def _actual_painter_order(self) -> tuple[str, ...]:
        expected_z = self.controller.active_painter_z_indices
        prepared = self.controller._last_prepared_frame
        if prepared is None:
            raise GlobalParallelRigError(
                "global controller has no committed painter frame"
            )
        item_mobjects = prepared.numeric.item_mobjects
        if set(item_mobjects) != set(expected_z):
            raise GlobalParallelRigError(
                "global painter cache differs from its committed item set"
            )
        signature = {
            item_id: (root_id, z_index)
            for item_id, root_id, z_index
            in self.controller._last_painter_band_signature
        }
        live: list[tuple[str, float]] = []
        for item_id, root in item_mobjects.items():
            if item_id not in signature or signature[item_id][0] != id(root):
                raise GlobalParallelRigError(
                    "global painter item identity drifted"
                )
            z_index = float(getattr(root, "z_index", float("nan")))
            if not np.isfinite(z_index) or z_index != expected_z[item_id]:
                raise GlobalParallelRigError(
                    "global painter z-index differs from its managed band"
                )
            if signature[item_id][1] != z_index:
                raise GlobalParallelRigError(
                    "global painter signature differs from live z-index"
                )
            live.append((item_id, z_index))
        return tuple(item_id for item_id, _z in sorted(live, key=lambda item: item[1]))

    def attach(self) -> "GlobalParallelRigBinding":
        if self.controller.attached:
            return self
        if any(item.controller.attached for item in self.bindings):
            raise GlobalParallelRigError(
                "local Rig controllers must never be attached to a global binding"
            )
        try:
            _require_semantic_scene_camera(self.scene)
        except ParallelSectionRigBindingError as exc:
            raise GlobalParallelRigError(
                f"shared scene.camera is not a valid parallel viewport: {exc}"
            ) from exc
        self._stage_index(0, projection_override=True)
        try:
            self.controller.attach()
            if self._actual_painter_order() != self.sequence.evidences[0].painter_order.draw_order:
                raise GlobalParallelRigError(
                    "attached global painter order differs from dry evidence"
                )
        except BaseException:
            if self.controller.attached:
                self.controller.restore()
            raise
        finally:
            self._projection_override = None
        return self

    def _gate_participant(self) -> ParallelFrameParticipant[ParallelFrameState]:
        by_id = {frame.frame_id: index for index, frame in enumerate(self.sequence.frames)}

        def prepare(frame: ParallelFrameState) -> GlobalParallelRigStageFrame:
            if not isinstance(frame, ParallelFrameState):
                raise TypeError("global playback requires ParallelFrameState")
            if any(item.controller.attached for item in self.bindings):
                raise GlobalParallelRigError(
                    "local Rig controllers must remain unattached during playback"
                )
            try:
                index = by_id[frame.frame_id]
            except KeyError as exc:
                raise GlobalParallelRigError("frame is absent from global preflight") from exc
            expected = self.sequence.frames[index]
            if (
                frame.preflight_input_digest
                != self.sequence.evidences[index].digest
                or frame.camera is not expected.camera
            ):
                raise GlobalParallelRigError(
                    "global frame differs from its dry-prepared evidence"
                )
            if tuple(frame.channels) != tuple(expected.channels):
                raise GlobalParallelRigError(
                    "global frame channels differ from preflight"
                )
            viewport_transform = frame.channel(
                PARALLEL_VIEWPORT_TRANSFORM_CHANNEL
            )
            expected_viewport_transform = expected.channel(
                PARALLEL_VIEWPORT_TRANSFORM_CHANNEL
            )
            if (
                not isinstance(viewport_transform, ParallelScreenTransform)
                or viewport_transform != expected_viewport_transform
            ):
                raise GlobalParallelRigError(
                    "global frame viewport transform differs from preflight"
                )
            stage = frame.channel(GLOBAL_PARALLEL_RIG_STAGE_CHANNEL)
            if stage is not expected.channel(GLOBAL_PARALLEL_RIG_STAGE_CHANNEL):
                raise GlobalParallelRigError(
                    "global frame stage payload differs from preflight"
                )
            assert isinstance(stage, GlobalParallelRigStageFrame)
            current = tuple(
                (rig_id, self._rig_state_digest(binding, index))
                for rig_id, binding in zip(self.sequence.rig_ids, self.bindings)
            )
            if current != stage.evidence.rig_state_digests:
                raise GlobalParallelRigError(
                    "local Rig source evidence changed after global compilation"
                )
            return stage

        return ParallelFrameParticipant(
            participant_id="global-parallel-preflight",
            phase=ParallelFramePhase.PREFLIGHT,
            prepare=prepare,
            snapshot=lambda: None,
            commit=lambda _value: None,
            rollback=lambda _value: None,
            binding_kind=ParallelFrameBindingKind.PREFLIGHT_GATE,
        )

    def _stage_participant(self) -> ParallelFrameParticipant[ParallelFrameState]:
        def prepare(frame: ParallelFrameState) -> GlobalParallelRigStageFrame:
            value = frame.channel(GLOBAL_PARALLEL_RIG_STAGE_CHANNEL)
            if not isinstance(value, GlobalParallelRigStageFrame):
                raise TypeError("global stage channel is invalid")
            return value

        def commit(value: object) -> None:
            if not isinstance(value, GlobalParallelRigStageFrame):
                raise TypeError("prepared global stage is invalid")
            self._stage_payload(value, projection_override=False)

        def rollback(value: object) -> None:
            if not isinstance(value, tuple):
                raise TypeError("global stage snapshot is invalid")
            self._restore_local_stage(value)

        return ParallelFrameParticipant(
            participant_id="global-parallel-provider-stage",
            phase=ParallelFramePhase.GEOMETRY,
            prepare=prepare,
            snapshot=self._capture_local_stage,
            commit=commit,
            rollback=rollback,
            binding_kind=ParallelFrameBindingKind.SECTION_BANK,
        )

    def _paint_participant(self) -> ParallelFrameParticipant[ParallelFrameState]:
        def prepare(frame: ParallelFrameState) -> GlobalParallelRigFrameEvidence:
            stage = frame.channel(GLOBAL_PARALLEL_RIG_STAGE_CHANNEL)
            if not isinstance(stage, GlobalParallelRigStageFrame):
                raise TypeError("global stage channel is invalid")
            return stage.evidence

        def commit(value: object) -> None:
            if not isinstance(value, GlobalParallelRigFrameEvidence):
                raise TypeError("global painter evidence is invalid")
            self.controller.update()
            if self._actual_painter_order() != value.painter_order.draw_order:
                raise GlobalParallelRigError(
                    "runtime global painter order differs from dry evidence"
                )

        return ParallelFrameParticipant(
            participant_id="global-parallel-paint",
            phase=ParallelFramePhase.PAINT,
            prepare=prepare,
            snapshot=self.controller.snapshot_transaction_state,
            commit=commit,
            rollback=self.controller.restore_transaction_state,
            binding_kind=ParallelFrameBindingKind.SECTION_DISPLAY,
        )

    def build_coordinator(
        self,
        camera: object,
    ) -> ParallelFrameCoordinator[ParallelFrameState]:
        if not self.controller.attached:
            raise GlobalParallelRigError(
                "attach the global binding before building its coordinator"
            )
        if camera is not getattr(self.scene, "camera", None):
            raise GlobalParallelRigError(
                "the playback camera must be the shared scene.camera"
            )
        if any(item.controller.attached for item in self.bindings):
            raise GlobalParallelRigError(
                "local Rig controllers must remain unattached during playback"
            )
        if self._coordinator is not None:
            if self._coordinator.poisoned:
                raise GlobalParallelRigError("global playback coordinator is poisoned")
            return self._coordinator
        coordinator: ParallelFrameCoordinator[ParallelFrameState] = (
            ParallelFrameCoordinator()
        )
        coordinator.add(self._gate_participant())
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
        coordinator.add(self._stage_participant())
        coordinator.add(self._paint_participant())
        self._coordinator = coordinator
        return coordinator

    def restore(self) -> "GlobalParallelRigBinding":
        if self._coordinator is not None and self._coordinator.active:
            self._coordinator.restore()
        if self.controller.attached:
            self.controller.restore()
        self._stage_index(0, projection_override=True)
        return self


def compile_global_parallel_rig(
    bindings: Sequence[ParallelSectionRigBinding],
) -> GlobalParallelRigBinding:
    """Compile global dry evidence without attaching any Scene object."""

    return GlobalParallelRigBinding(bindings)


__all__ = [
    "GLOBAL_PARALLEL_RIG_SCHEMA",
    "GLOBAL_PARALLEL_RIG_STAGE_CHANNEL",
    "GlobalParallelRigBinding",
    "GlobalParallelRigError",
    "GlobalParallelRigFrameEvidence",
    "GlobalParallelRigSequence",
    "GlobalParallelRigStageFrame",
    "compile_global_parallel_rig",
]
