from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np

from tikz_native.parallel_camera import ParallelCameraState
from tikz_native.parallel_frame import (
    ParallelFrameCoordinator,
    ParallelFrameCoordinatorError,
    ParallelFrameParticipant,
    ParallelFramePhase,
    ParallelFrameState,
)
from tikz_native.parallel_preflight import (
    PARALLEL_PREFLIGHT_REPORT_SCHEMA,
    PARALLEL_PREFLIGHT_FRAME_CHANNEL,
    CapacityEvidence,
    PainterOrderEvidence,
    ParallelPreflightError,
    ParallelPreflightFrame,
    ParallelPreflightGate,
    ParallelPreflightLimits,
    ParallelPreflightRejectedError,
    ParallelSafeFrame,
    ParallelScreenTransform,
    TopologyEventEvidence,
    preflight_parallel_frames,
)


def _camera(*, zoom: float = 1.0) -> ParallelCameraState:
    return ParallelCameraState(
        np.identity(3),
        target=(1.0, 2.0, 3.0),
        screen_anchor=(0.25, -0.1),
        zoom=zoom,
    )


def _limits(**overrides: object) -> ParallelPreflightLimits:
    values: dict[str, object] = {
        "safe_frame": ParallelSafeFrame(-2.0, 2.0, -1.0, 1.0),
        "min_zoom": 0.5,
        "max_zoom": 2.0,
        "tolerance": 1.0e-9,
        "require_framing_points": True,
    }
    values.update(overrides)
    return ParallelPreflightLimits(**values)  # type: ignore[arg-type]


def _painter() -> PainterOrderEvidence:
    return PainterOrderEvidence(
        item_ids=("surface", "plane", "curve"),
        relations=(("surface", "plane"), ("plane", "curve")),
        draw_order=("surface", "plane", "curve"),
    )


def _frame(
    frame_id: str = "frame-0",
    time: float = 0.0,
    **overrides: object,
) -> ParallelPreflightFrame:
    values: dict[str, object] = {
        "frame_id": frame_id,
        "time": time,
        "camera": _camera(),
        "framing_points": (
            (1.0, 2.0, 3.0),
            (2.0, 2.5, 3.0),
            (0.0, 1.5, 3.0),
        ),
        "topology_events": (
            TopologyEventEvidence(
                "event-0",
                "ellipse-to-parabola",
                True,
                requires_slot_bank=True,
                slot_bank_id="conic-bank-1",
            ),
        ),
        "capacities": (
            CapacityEvidence("conic-bank-1", 4, 8),
            CapacityEvidence("plane-slots", 10, 10),
        ),
        "painter_order": _painter(),
    }
    values.update(overrides)
    return ParallelPreflightFrame(**values)  # type: ignore[arg-type]


def _codes(report: object) -> set[str]:
    return {item.code for item in report.issues}  # type: ignore[attr-defined]


def _text_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()


class TikzNativeParallelPreflightTests(unittest.TestCase):
    def test_complete_joint_evidence_is_accepted_and_counted(self) -> None:
        first = _frame()
        second = _frame(
            "frame-1",
            1.0,
            topology_events=(
                TopologyEventEvidence(
                    "event-1",
                    "plane-rank-one",
                    True,
                    requires_slot_bank=False,
                ),
            ),
        )

        report = preflight_parallel_frames((first, second), _limits())

        self.assertTrue(report.accepted)
        self.assertIs(report.require_accepted(), report)
        self.assertEqual(report.frame_count, 2)
        self.assertEqual(report.framing_point_count, 6)
        self.assertEqual(report.topology_event_count, 2)
        self.assertEqual(report.capacity_count, 4)
        self.assertEqual(report.painter_relation_count, 4)
        self.assertEqual(report.issues, ())

    def test_camera_target_anchor_and_zoom_are_used_for_safe_frame(self) -> None:
        target = (10.0, -4.0, 2.0)
        state = ParallelCameraState(
            np.identity(3),
            target=target,
            screen_anchor=(1.5, 0.75),
            zoom=2.0,
        )
        accepted = _frame(
            camera=state,
            framing_points=(target, (10.2, -4.2, 2.0)),
        )
        rejected = _frame(
            camera=state,
            framing_points=(target, (10.3, -4.2, 2.0)),
        )

        self.assertTrue(preflight_parallel_frames((accepted,), _limits()).accepted)
        report = preflight_parallel_frames((rejected,), _limits())
        self.assertIn("safe-frame-overflow", _codes(report))

    def test_renderer_zoom_center_and_display_offset_are_preflighted(self) -> None:
        state = ParallelCameraState(
            np.identity(3),
            target=(1.0, 2.0, 3.0),
            screen_anchor=(0.5, -0.25),
            zoom=1.25,
        )
        transform = ParallelScreenTransform(
            inherited_zoom=1.6,
            frame_center=(0.2, -0.1),
            display_offset=(-0.15, 0.05),
        )
        point = (1.5, 2.25, 3.0)
        # anchor + inherited_zoom * state.zoom * delta + center + offset
        expected = np.asarray((1.55, 0.2))
        projected = transform.apply(
            state.project_points((point,)),
            state.screen_anchor,
        )
        np.testing.assert_allclose(projected[0], expected, atol=1.0e-12, rtol=0.0)

        frame = _frame(
            camera=state,
            screen_transform=transform,
            framing_points=(point,),
        )
        self.assertTrue(preflight_parallel_frames((frame,), _limits()).accepted)
        too_large = _frame(
            camera=state,
            screen_transform=ParallelScreenTransform(inherited_zoom=1.7),
            framing_points=((1.0, 2.0, 3.0),),
        )
        self.assertIn(
            "zoom-above-maximum",
            _codes(preflight_parallel_frames((too_large,), _limits())),
        )

    def test_safe_frame_boundary_uses_explicit_tolerance(self) -> None:
        state = ParallelCameraState(np.identity(3))
        boundary = _frame(
            camera=state,
            framing_points=((2.0 + 0.5e-9, 1.0, 0.0),),
        )
        outside = _frame(
            camera=state,
            framing_points=((2.0 + 2.0e-9, 1.0, 0.0),),
        )

        self.assertTrue(preflight_parallel_frames((boundary,), _limits()).accepted)
        self.assertIn(
            "safe-frame-overflow",
            _codes(preflight_parallel_frames((outside,), _limits())),
        )

    def test_zoom_range_and_missing_framing_points_fail_closed(self) -> None:
        target_only = ((1.0, 2.0, 3.0),)
        below = _frame(
            "below",
            0.0,
            camera=_camera(zoom=0.4),
            framing_points=target_only,
        )
        above = _frame(
            "above",
            1.0,
            camera=_camera(zoom=2.1),
            framing_points=target_only,
        )
        missing = _frame("missing", 2.0, framing_points=())

        report = preflight_parallel_frames((below, above, missing), _limits())

        self.assertEqual(
            _codes(report),
            {
                "zoom-below-minimum",
                "zoom-above-maximum",
                "missing-framing-points",
                "duplicate-topology-event",
            },
        )
        optional = preflight_parallel_frames(
            (_frame(framing_points=()),),
            _limits(require_framing_points=False),
        )
        self.assertTrue(optional.accepted)

    def test_topology_events_require_certification_and_declared_banks(self) -> None:
        events = (
            TopologyEventEvidence("unknown", "numeric-root", False),
            TopologyEventEvidence(
                "handoff",
                "parabola-to-hyperbola",
                True,
                requires_slot_bank=True,
            ),
        )
        report = preflight_parallel_frames(
            (_frame(topology_events=events),),
            _limits(),
        )

        self.assertEqual(
            _codes(report),
            {"uncertified-topology-event", "missing-topology-slot-bank"},
        )

        accepted = TopologyEventEvidence(
            "handoff-with-bank",
            "parabola-to-hyperbola",
            True,
            requires_slot_bank=True,
            slot_bank_id="conic-bank-1",
        )
        accepted_report = preflight_parallel_frames(
            (_frame(topology_events=(accepted,)),),
            _limits(),
        )
        self.assertTrue(accepted_report.accepted)
        self.assertEqual(accepted_report.topology_event_count, 1)

    def test_topology_bank_and_fixed_capacity_are_linked_across_frames(self) -> None:
        first = _frame()
        second = _frame(
            "frame-1",
            1.0,
            topology_events=(
                TopologyEventEvidence(
                    "event-1",
                    "parabola-to-hyperbola",
                    True,
                    requires_slot_bank=True,
                    slot_bank_id="missing-bank",
                ),
            ),
            capacities=(
                CapacityEvidence("conic-bank-1", 1, 4),
                CapacityEvidence("new-resource", 1, 1),
            ),
        )
        report = preflight_parallel_frames((first, second), _limits())
        self.assertEqual(
            _codes(report),
            {
                "unknown-topology-slot-bank",
                "capacity-resource-set-changed",
                "capacity-limit-changed",
            },
        )

    def test_capacity_overflow_negative_and_duplicate_resources_are_reported(
        self,
    ) -> None:
        capacities = (
            CapacityEvidence("curves", 9, 8),
            CapacityEvidence("curves", 4, 8),
            CapacityEvidence("dashes", -1, 4),
        )
        report = preflight_parallel_frames(
            (_frame(capacities=capacities, topology_events=()),),
            _limits(),
        )

        self.assertEqual(
            _codes(report),
            {
                "capacity-overflow",
                "duplicate-capacity-resource",
                "negative-capacity",
            },
        )

    def test_painter_order_checks_items_relations_order_and_cycles(self) -> None:
        evidence = PainterOrderEvidence(
            item_ids=("a", "a", "b", "c"),
            relations=(
                ("a", "b"),
                ("a", "b"),
                ("b", "a"),
                ("c", "c"),
                ("c", "missing"),
            ),
            draw_order=("b", "a", "a", "extra"),
        )
        report = preflight_parallel_frames(
            (_frame(painter_order=evidence),),
            _limits(),
        )

        self.assertEqual(
            _codes(report),
            {
                "duplicate-painter-item",
                "duplicate-draw-order-item",
                "draw-order-item-mismatch",
                "duplicate-painter-relation",
                "self-painter-relation",
                "unknown-painter-relation-item",
                "painter-relation-cycle",
            },
        )

        wrong_order = PainterOrderEvidence(
            item_ids=("a", "b"),
            relations=(("a", "b"),),
            draw_order=("b", "a"),
        )
        report = preflight_parallel_frames(
            (_frame(painter_order=wrong_order),),
            _limits(),
        )
        self.assertEqual(_codes(report), {"draw-order-violates-relation"})

    def test_sequence_identity_and_time_are_audited(self) -> None:
        first = _frame("same", 1.0, topology_events=())
        second = _frame("same", 1.0, topology_events=())
        report = preflight_parallel_frames((first, second), _limits())

        self.assertEqual(
            _codes(report),
            {"duplicate-frame-id", "frame-time-not-increasing"},
        )
        empty = preflight_parallel_frames((), _limits())
        self.assertIn("empty-sequence", _codes(empty))

    def test_rejected_report_raises_one_concise_authoring_error(self) -> None:
        report = preflight_parallel_frames(
            (_frame(camera=_camera(zoom=3.0), framing_points=()),),
            _limits(),
        )
        self.assertFalse(report.accepted)
        with self.assertRaisesRegex(
            ParallelPreflightRejectedError,
            r"rejected 2 error\(s\)",
        ):
            report.require_accepted()

    def test_report_json_and_digest_are_deterministic(self) -> None:
        frame = _frame()
        first = preflight_parallel_frames((frame,), _limits())
        second = preflight_parallel_frames((frame,), _limits())
        equivalent = _frame(
            painter_order=PainterOrderEvidence(
                item_ids=("curve", "plane", "surface"),
                relations=(("plane", "curve"), ("surface", "plane")),
                draw_order=("surface", "plane", "curve"),
            )
        )
        third = preflight_parallel_frames((equivalent,), _limits())

        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(first.input_digest, second.input_digest)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.input_digest, third.input_digest)
        payload = json.loads(first.to_json())
        self.assertEqual(payload["schema"], PARALLEL_PREFLIGHT_REPORT_SCHEMA)
        self.assertTrue(payload["accepted"])
        self.assertEqual(payload["frameIds"], ["frame-0"])
        self.assertEqual(payload["frameDigests"], [frame.digest])

    def test_accepted_report_gates_exact_frame_sequence_and_rolls_back(self) -> None:
        first = _frame()
        second = _frame(
            "frame-1",
            1.0,
            topology_events=(
                TopologyEventEvidence(
                    "event-1",
                    "parabola-to-hyperbola",
                    True,
                    requires_slot_bank=True,
                    slot_bank_id="conic-bank-1",
                ),
            ),
        )
        report = preflight_parallel_frames((first, second), _limits())

        mismatch_gate = ParallelPreflightGate(report)
        mismatch_coordinator: ParallelFrameCoordinator[ParallelFrameState]
        mismatch_coordinator = ParallelFrameCoordinator()
        mismatch_coordinator.add(mismatch_gate.participant())
        with self.assertRaisesRegex(
            ParallelFrameCoordinatorError,
            "camera differs",
        ):
            mismatch_coordinator.update(
                ParallelFrameState(
                    _camera(zoom=1.1),
                    channels={PARALLEL_PREFLIGHT_FRAME_CHANNEL: first},
                    frame_id=first.frame_id,
                    preflight_input_digest=report.input_digest,
                )
            )

        gate = ParallelPreflightGate(report)
        coordinator: ParallelFrameCoordinator[ParallelFrameState]
        coordinator = ParallelFrameCoordinator()
        coordinator.add(gate.participant())
        coordinator.update(
            ParallelFrameState(
                first.camera,
                channels={PARALLEL_PREFLIGHT_FRAME_CHANNEL: first},
                frame_id=first.frame_id,
                preflight_input_digest=report.input_digest,
            )
        )
        self.assertEqual(gate.next_frame_index, 1)

        with self.assertRaisesRegex(
            ParallelFrameCoordinatorError,
            "does not belong",
        ):
            coordinator.update(
                ParallelFrameState(
                    second.camera,
                    channels={PARALLEL_PREFLIGHT_FRAME_CHANNEL: second},
                    frame_id=second.frame_id,
                    preflight_input_digest="sha256:" + "0" * 64,
                )
            )
        self.assertEqual(gate.next_frame_index, 1)
        coordinator.update(
            ParallelFrameState(
                second.camera,
                channels={PARALLEL_PREFLIGHT_FRAME_CHANNEL: second},
                frame_id=second.frame_id,
                preflight_input_digest=report.input_digest,
            )
        )
        self.assertEqual(gate.next_frame_index, 2)
        coordinator.restore()
        self.assertEqual(gate.next_frame_index, 0)

        failing_gate = ParallelPreflightGate(report)
        failing_coordinator: ParallelFrameCoordinator[ParallelFrameState]
        failing_coordinator = ParallelFrameCoordinator()
        failing_coordinator.add(failing_gate.participant())

        def fail_commit(_prepared: object) -> None:
            raise RuntimeError("synthetic commit failure")

        failing_coordinator.add(
            ParallelFrameParticipant(
                "synthetic-failure",
                ParallelFramePhase.PAINT,
                lambda _frame: None,
                lambda: None,
                fail_commit,
                lambda _snapshot: None,
            )
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic commit failure"):
            failing_coordinator.update(
                ParallelFrameState(
                    first.camera,
                    channels={PARALLEL_PREFLIGHT_FRAME_CHANNEL: first},
                    frame_id=first.frame_id,
                    preflight_input_digest=report.input_digest,
                )
            )
        self.assertEqual(failing_gate.next_frame_index, 0)

        rejected = preflight_parallel_frames((), _limits())
        with self.assertRaises(ParallelPreflightRejectedError):
            ParallelPreflightGate(rejected)

    def test_gate_binds_declared_runtime_channel_digests(self) -> None:
        expected = "certified-plane"
        frame = _frame(
            channel_digests=(("section-plane", _text_digest(expected)),),
        )
        report = preflight_parallel_frames((frame,), _limits())
        gate = ParallelPreflightGate(
            report,
            channel_digesters={"section-plane": _text_digest},
        )
        coordinator: ParallelFrameCoordinator[ParallelFrameState]
        coordinator = ParallelFrameCoordinator()
        coordinator.add(gate.participant())
        with self.assertRaisesRegex(
            ParallelFrameCoordinatorError,
            "section-plane.*differs",
        ):
            coordinator.update(
                ParallelFrameState(
                    frame.camera,
                    channels={
                        PARALLEL_PREFLIGHT_FRAME_CHANNEL: frame,
                        "section-plane": "tampered-plane",
                    },
                    frame_id=frame.frame_id,
                    preflight_input_digest=report.input_digest,
                )
            )
        self.assertEqual(gate.next_frame_index, 0)
        coordinator.update(
            ParallelFrameState(
                frame.camera,
                channels={
                    PARALLEL_PREFLIGHT_FRAME_CHANNEL: frame,
                    "section-plane": expected,
                },
                frame_id=frame.frame_id,
                preflight_input_digest=report.input_digest,
            )
        )
        self.assertEqual(gate.next_frame_index, 1)

        extra_gate = ParallelPreflightGate(
            report,
            channel_digesters={"section-plane": _text_digest},
        )
        extra_coordinator: ParallelFrameCoordinator[ParallelFrameState]
        extra_coordinator = ParallelFrameCoordinator()
        extra_coordinator.add(extra_gate.participant())
        with self.assertRaisesRegex(
            ParallelFrameCoordinatorError,
            "runtime channels differ.*unpreflighted-command",
        ):
            extra_coordinator.update(
                ParallelFrameState(
                    frame.camera,
                    channels={
                        PARALLEL_PREFLIGHT_FRAME_CHANNEL: frame,
                        "section-plane": expected,
                        "unpreflighted-command": "mutate",
                    },
                    frame_id=frame.frame_id,
                    preflight_input_digest=report.input_digest,
                )
            )

        missing = ParallelPreflightGate(report)
        missing_coordinator: ParallelFrameCoordinator[ParallelFrameState]
        missing_coordinator = ParallelFrameCoordinator()
        missing_coordinator.add(missing.participant())
        with self.assertRaisesRegex(
            ParallelFrameCoordinatorError,
            "no canonical digester",
        ):
            missing_coordinator.update(
                ParallelFrameState(
                    frame.camera,
                    channels={
                        PARALLEL_PREFLIGHT_FRAME_CHANNEL: frame,
                        "section-plane": expected,
                    },
                    frame_id=frame.frame_id,
                    preflight_input_digest=report.input_digest,
                )
            )

    def test_malformed_evidence_is_rejected_before_preflight(self) -> None:
        with self.assertRaisesRegex(ParallelPreflightError, "positive width"):
            ParallelSafeFrame(1.0, 1.0, -1.0, 1.0)
        with self.assertRaisesRegex(ParallelPreflightError, "must not exceed"):
            _limits(min_zoom=2.0, max_zoom=1.0)
        with self.assertRaisesRegex(ParallelPreflightError, "must be an integer"):
            CapacityEvidence("curves", True, 4)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ParallelPreflightError, "two item ids"):
            PainterOrderEvidence(
                item_ids=("a",),
                relations=(("a",),),  # type: ignore[arg-type]
                draw_order=("a",),
            )
        with self.assertRaisesRegex(ParallelPreflightError, "three finite"):
            _frame(framing_points=((0.0, float("nan"), 0.0),))


if __name__ == "__main__":
    unittest.main()
