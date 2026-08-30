from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json
import unittest

import numpy as np

from polyhedron_visibility.quadrics.section_timeline_transition import (
    SectionTimelineLayerRole,
)
from tikz_native.parallel_camera import ParallelCameraState
from tikz_native.parallel_frame import (
    ParallelFrameCoordinator,
    ParallelFrameParticipant,
    ParallelFramePhase,
    ParallelFrameState,
)
from tikz_native.section_bank_render import (
    SECTION_BANK_RENDER_CHANNEL,
    SECTION_BANK_RENDER_FRAME_SCHEMA,
    SectionBankRenderError,
    SectionBankRenderFrame,
    SectionBankRenderLayer,
    section_bank_frame_participant,
)


def _layer(
    bank_index: int,
    *,
    opacity: float,
    role: SectionTimelineLayerRole,
    reference_frame_index: int | None = None,
    active_cap_chord_ids: tuple[str, ...] = (),
) -> SectionBankRenderLayer:
    reference = (
        bank_index
        if reference_frame_index is None
        else reference_frame_index
    )
    digest = "sha256:" + hashlib.sha256(
        f"geometry-{bank_index}-{reference}".encode("utf-8")
    ).hexdigest()
    return SectionBankRenderLayer(
        bank_index=bank_index,
        semantic_bank_id=f" semantic-bank-{bank_index} ",
        reference_frame_index=reference,
        geometry_time=0.5 + reference,
        opacity=opacity,
        branch_count=bank_index + 1,
        isolated_point_count=bank_index,
        role=role,
        geometry_digest=digest,
        active_cap_chord_ids=active_cap_chord_ids,
    )


def _crossfade_frame() -> SectionBankRenderFrame:
    return SectionBankRenderFrame(
        1.25,
        (
            _layer(
                1,
                opacity=0.35,
                role=SectionTimelineLayerRole.EXACT_CRITICAL,
                reference_frame_index=4,
                active_cap_chord_ids=("cap:z",),
            ),
            _layer(
                0,
                opacity=0.65,
                role=SectionTimelineLayerRole.LIVE_BEFORE,
                reference_frame_index=3,
                active_cap_chord_ids=("cap:a",),
            ),
        ),
        (" cap:z ", "cap:a"),
    )


class SectionBankRenderContractTests(unittest.TestCase):
    def test_frame_is_immutable_canonical_and_digest_stable(self) -> None:
        first = _crossfade_frame()
        second = SectionBankRenderFrame(
            first.time,
            tuple(reversed(first.layers)),
            tuple(reversed(first.active_cap_chord_ids)),
        )

        self.assertEqual(tuple(item.bank_index for item in first.layers), (0, 1))
        self.assertEqual(
            first.active_cap_chord_ids,
            ("cap:a", "cap:z"),
        )
        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(len(first.digest), len("sha256:") + 64)
        self.assertEqual(first.layers[0].digest, second.layers[0].digest)
        self.assertEqual(
            len(first.layers[0].digest),
            len("sha256:") + 64,
        )

        payload = json.loads(first.to_json())
        self.assertEqual(payload["schema"], SECTION_BANK_RENDER_FRAME_SCHEMA)
        self.assertEqual(
            [item["bankIndex"] for item in payload["layers"]],
            [0, 1],
        )
        self.assertEqual(payload["activeCapChordIds"], ["cap:a", "cap:z"])
        with self.assertRaises(FrozenInstanceError):
            first.time = 3.0  # type: ignore[misc]

    def test_layer_rejects_invalid_scalar_and_role_values(self) -> None:
        valid = {
            "bank_index": 0,
            "semantic_bank_id": "bank-a",
            "reference_frame_index": 2,
            "geometry_time": 0.5,
            "opacity": 1.0,
            "branch_count": 1,
            "isolated_point_count": 0,
            "role": SectionTimelineLayerRole.LIVE,
            "geometry_digest": "sha256:" + "0" * 64,
        }
        invalid = (
            ("bank_index", True, "non-negative integer"),
            ("bank_index", 2, "0 or 1"),
            ("semantic_bank_id", " ", "non-empty"),
            ("reference_frame_index", -1, "non-negative integer"),
            ("geometry_time", float("nan"), "finite"),
            ("opacity", 1.1, r"\[0, 1\]"),
            ("branch_count", True, "non-negative integer"),
            ("isolated_point_count", True, "non-negative integer"),
            ("role", "unknown", "SectionTimelineLayerRole"),
            ("geometry_digest", "sha256:bad", "lowercase sha256"),
        )
        for field, value, message in invalid:
            with self.subTest(field=field, value=value):
                candidate = dict(valid)
                candidate[field] = value
                with self.assertRaisesRegex(SectionBankRenderError, message):
                    SectionBankRenderLayer(**candidate)  # type: ignore[arg-type]

    def test_frame_rejects_ambiguous_banks_opacities_roles_and_caps(self) -> None:
        live = _layer(
            0,
            opacity=1.0,
            role=SectionTimelineLayerRole.LIVE,
        )
        with self.assertRaisesRegex(
            SectionBankRenderError,
            "one or two",
        ):
            SectionBankRenderFrame(0.0, ())
        with self.assertRaisesRegex(TypeError, "SectionBankRenderLayer"):
            SectionBankRenderFrame(0.0, (object(),))  # type: ignore[arg-type]
        with self.assertRaisesRegex(
            SectionBankRenderError,
            "unique bank indices",
        ):
            SectionBankRenderFrame(
                0.0,
                (
                    _layer(
                        0,
                        opacity=0.5,
                        role=SectionTimelineLayerRole.LIVE_BEFORE,
                    ),
                    SectionBankRenderLayer(
                        bank_index=0,
                        semantic_bank_id="other-bank",
                        reference_frame_index=1,
                        geometry_time=1.0,
                        opacity=0.5,
                        branch_count=1,
                        isolated_point_count=0,
                        role=SectionTimelineLayerRole.EXACT_CRITICAL,
                        geometry_digest="sha256:" + "1" * 64,
                    ),
                ),
            )
        with self.assertRaisesRegex(
            SectionBankRenderError,
            "unique semantic bank ids",
        ):
            SectionBankRenderFrame(
                0.0,
                (
                    _layer(
                        0,
                        opacity=0.5,
                        role=SectionTimelineLayerRole.LIVE_BEFORE,
                    ),
                    SectionBankRenderLayer(
                        bank_index=1,
                        semantic_bank_id="semantic-bank-0",
                        reference_frame_index=1,
                        geometry_time=1.0,
                        opacity=0.5,
                        branch_count=1,
                        isolated_point_count=0,
                        role=SectionTimelineLayerRole.EXACT_CRITICAL,
                        geometry_digest="sha256:" + "1" * 64,
                    ),
                ),
            )
        with self.assertRaisesRegex(SectionBankRenderError, "sum to one"):
            SectionBankRenderFrame(
                0.0,
                (
                    _layer(
                        0,
                        opacity=0.4,
                        role=SectionTimelineLayerRole.LIVE_BEFORE,
                    ),
                    _layer(
                        1,
                        opacity=0.4,
                        role=SectionTimelineLayerRole.EXACT_CRITICAL,
                    ),
                ),
            )
        with self.assertRaisesRegex(SectionBankRenderError, "positive active"):
            SectionBankRenderFrame(
                0.0,
                (
                    _layer(
                        0,
                        opacity=0.0,
                        role=SectionTimelineLayerRole.LIVE_BEFORE,
                    ),
                    _layer(
                        1,
                        opacity=1.0,
                        role=SectionTimelineLayerRole.EXACT_CRITICAL,
                    ),
                ),
            )
        with self.assertRaisesRegex(
            SectionBankRenderError,
            "certified crossfade side",
        ):
            SectionBankRenderFrame(
                0.0,
                (
                    _layer(
                        0,
                        opacity=0.5,
                        role=SectionTimelineLayerRole.LIVE_BEFORE,
                    ),
                    _layer(
                        1,
                        opacity=0.5,
                        role=SectionTimelineLayerRole.LIVE_AFTER,
                    ),
                ),
            )
        with self.assertRaisesRegex(
            SectionBankRenderError,
            "single bank layer",
        ):
            SectionBankRenderFrame(
                0.0,
                (
                    _layer(
                        0,
                        opacity=1.0,
                        role=SectionTimelineLayerRole.LIVE_BEFORE,
                    ),
                ),
            )
        with self.assertRaisesRegex(SectionBankRenderError, "must be unique"):
            SectionBankRenderFrame(
                0.0,
                (live,),
                ("cap-a", " cap-a "),
            )
        with self.assertRaisesRegex(TypeError, "sequence of string ids"):
            SectionBankRenderFrame(
                0.0,
                (live,),
                "cap-a",  # type: ignore[arg-type]
            )


class _BankTarget:
    def __init__(self) -> None:
        self.banks = {
            "semantic-bank-0": ("baseline-a", 1.0),
            "semantic-bank-1": ("baseline-b", 0.0),
        }
        self.active_cap_chord_ids = ("baseline-cap",)
        self.applied_layers: list[str] = []
        self.fail_on_second_layer = False

    def snapshot_section_bank_render_state(self) -> object:
        return (
            deepcopy(self.banks),
            self.active_cap_chord_ids,
            tuple(self.applied_layers),
        )

    def apply_section_bank_render_frame(
        self,
        frame: SectionBankRenderFrame,
    ) -> None:
        for index, layer in enumerate(frame.layers):
            self.banks[layer.semantic_bank_id] = (
                layer.reference_frame_index,
                layer.geometry_time,
                layer.opacity,
                layer.branch_count,
                layer.isolated_point_count,
                layer.role.value,
                layer.geometry_digest,
                layer.active_cap_chord_ids,
            )
            self.applied_layers.append(layer.semantic_bank_id)
            if self.fail_on_second_layer and index == 1:
                raise RuntimeError("synthetic second-bank apply failure")
        self.active_cap_chord_ids = frame.active_cap_chord_ids

    def restore_section_bank_render_state(self, value: object) -> None:
        banks, cap_ids, applied = value  # type: ignore[misc]
        self.banks = deepcopy(banks)
        self.active_cap_chord_ids = cap_ids
        self.applied_layers = list(applied)


class SectionBankRenderParticipantTests(unittest.TestCase):
    def test_participant_commits_and_restores_one_full_frame(self) -> None:
        target = _BankTarget()
        baseline = target.snapshot_section_bank_render_state()
        frame = _crossfade_frame()
        coordinator: ParallelFrameCoordinator[ParallelFrameState]
        coordinator = ParallelFrameCoordinator()
        participant = section_bank_frame_participant(target)
        self.assertIs(participant.phase, ParallelFramePhase.GEOMETRY)
        coordinator.add(participant)

        coordinator.update(
            ParallelFrameState(
                ParallelCameraState(np.identity(3)),
                {SECTION_BANK_RENDER_CHANNEL: frame},
            )
        )

        self.assertEqual(
            target.applied_layers,
            ["semantic-bank-0", "semantic-bank-1"],
        )
        self.assertEqual(target.active_cap_chord_ids, ("cap:a", "cap:z"))
        coordinator.restore()
        self.assertEqual(target.snapshot_section_bank_render_state(), baseline)

    def test_second_bank_failure_rolls_back_target_and_prior_participant(
        self,
    ) -> None:
        target = _BankTarget()
        target.fail_on_second_layer = True
        baseline = target.snapshot_section_bank_render_state()
        prior_state = {"value": "baseline"}

        def prior_snapshot() -> object:
            return prior_state["value"]

        def prior_commit(_prepared: object) -> None:
            prior_state["value"] = "camera-committed"

        def prior_restore(value: object) -> None:
            prior_state["value"] = str(value)

        coordinator: ParallelFrameCoordinator[ParallelFrameState]
        coordinator = ParallelFrameCoordinator()
        coordinator.add(
            ParallelFrameParticipant(
                "prior-camera",
                ParallelFramePhase.CAMERA,
                lambda _frame: None,
                prior_snapshot,
                prior_commit,
                prior_restore,
            )
        )
        coordinator.add(section_bank_frame_participant(target))

        with self.assertRaisesRegex(
            RuntimeError,
            "second-bank apply failure",
        ):
            coordinator.update(
                ParallelFrameState(
                    ParallelCameraState(np.identity(3)),
                    {SECTION_BANK_RENDER_CHANNEL: _crossfade_frame()},
                )
            )

        self.assertEqual(prior_state["value"], "baseline")
        self.assertEqual(target.snapshot_section_bank_render_state(), baseline)
        self.assertIsNone(coordinator.last_committed_frame)
        self.assertFalse(coordinator.active)
        self.assertFalse(coordinator.poisoned)

    def test_participant_rejects_missing_protocol_and_wrong_channel(self) -> None:
        with self.assertRaisesRegex(TypeError, "snapshot, apply, and restore"):
            section_bank_frame_participant(object())

        coordinator: ParallelFrameCoordinator[ParallelFrameState]
        coordinator = ParallelFrameCoordinator()
        coordinator.add(section_bank_frame_participant(_BankTarget()))
        with self.assertRaisesRegex(
            TypeError,
            "SectionBankRenderFrame",
        ):
            coordinator.update(
                ParallelFrameState(
                    ParallelCameraState(np.identity(3)),
                    {SECTION_BANK_RENDER_CHANNEL: object()},
                )
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
