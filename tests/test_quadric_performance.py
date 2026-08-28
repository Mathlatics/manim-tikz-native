from __future__ import annotations

from dataclasses import replace
import json
from math import pi
import os
import unittest
from unittest.mock import patch

from manim import Scene, tempconfig

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.composite_authoring import (
    CompositeQuadricSection3D,
)
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.performance import (
    QUADRIC_PERFORMANCE_TRACE_ENV,
    QUADRIC_PERFORMANCE_TRACE_SCHEMA,
)
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section_boundary_curves,
    section_cap_chord_curve_ids,
)


VIEW = ParallelView.from_matrix(
    (
        (-0.7071067811865476, 0.7071067811865476, 0.0),
        (-0.4082482904638631, -0.4082482904638631, 0.8164965809277261),
        (0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
    )
)
SIDE_VIEW = ParallelView.from_matrix(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    )
)


def _limits(**overrides: object) -> QuadricManimLimits:
    values: dict[str, object] = {
        "max_surfaces": 2,
        "max_curves": 8,
        "max_fragments_per_curve": 12,
        "max_segments_per_fragment": 256,
        "max_surface_segments": 512,
        "max_dashes_per_fragment": 48,
        "max_projected_length": 18.0,
        "max_total_mobjects": 12000,
        "max_boundary_sources": 32,
    }
    values.update(overrides)
    return QuadricManimLimits(**values)  # type: ignore[arg-type]


def _single_controller(scene: Scene) -> QuadricOcclusion3D:
    cone = ConeSpec(
        "perf-cone",
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 3.0),
        radial_axis=(1.0, 0.0, 0.0),
        model=ConeModel.CLOSED_SINGLE,
    )
    plane = SectionPlane(
        "perf-plane",
        (0.0, 0.0, 0.2),
        (0.7, 0.0, 1.0),
        u_axis=(0.0, 1.0, 0.0),
    )
    section_id = "perf-section"
    curves = compute_quadric_section_boundary_curves(section_id, cone, plane)
    allocated = tuple(
        sorted(
            {
                *(item.curve_id for item in curves),
                *section_cap_chord_curve_ids(section_id, cone),
            }
        )
    )
    return QuadricOcclusion3D(
        scene,
        surfaces=(cone,),
        curves=curves,
        allocated_curve_ids=allocated,
        projection=VIEW,
        paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
        style=QuadricManimStyle(dash_length=0.3, dash_gap=0.2),
        limits=_limits(),
        section_plane=plane,
        boundary_visibility_mode="unified",
        include_surface_boundaries=True,
    )


class QuadricPerformanceTraceTests(unittest.TestCase):
    def test_tracing_is_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {QUADRIC_PERFORMANCE_TRACE_ENV: ""}):
            with tempconfig({"renderer": "cairo"}):
                controller = _single_controller(Scene()).attach()
                self.assertIsNone(controller.performance_snapshot())
                controller.restore()

    def test_identical_frame_skips_slot_writes_and_transaction_snapshots(
        self,
    ) -> None:
        with patch.dict(os.environ, {QUADRIC_PERFORMANCE_TRACE_ENV: "1"}):
            with tempconfig({"renderer": "cairo"}):
                controller = _single_controller(Scene()).attach()
                controller.update()
                snapshot = controller.performance_snapshot()
                self.assertIsNotNone(snapshot)
                assert snapshot is not None
                self.assertGreater(snapshot.counts["display_active_slot_count"], 0)
                self.assertEqual(snapshot.counts["display_changed_slot_count"], 0)
                self.assertEqual(snapshot.counts["display_hidden_slot_count"], 0)
                self.assertEqual(snapshot.counts["painter_band_changed_count"], 0)
                self.assertEqual(snapshot.counts["mutation_target_root_count"], 0)
                self.assertEqual(
                    snapshot.counts["transaction_snapshot_mobject_count"],
                    0,
                )
                self.assertEqual(snapshot.counts["modified_mobject_count"], 0)
                controller.restore()

    def test_changed_frame_snapshots_only_active_mutation_families(self) -> None:
        state = {"x": 0.0}

        def surfaces() -> tuple[SphereSpec, ...]:
            return (SphereSpec("moving-sphere", (state["x"], 0.0, 0.0), 1.0),)

        with patch.dict(os.environ, {QUADRIC_PERFORMANCE_TRACE_ENV: "1"}):
            with tempconfig({"renderer": "cairo"}):
                controller = QuadricOcclusion3D(
                    Scene(),
                    surfaces=surfaces,
                    curves=(),
                    projection=VIEW,
                    limits=_limits(),
                ).attach()
                state["x"] = 0.25
                controller.update()
                snapshot = controller.performance_snapshot()
                self.assertIsNotNone(snapshot)
                assert snapshot is not None
                self.assertGreater(snapshot.counts["display_changed_slot_count"], 0)
                self.assertEqual(snapshot.counts["display_hidden_slot_count"], 0)
                self.assertLess(
                    snapshot.counts["transaction_snapshot_mobject_count"],
                    snapshot.counts["mobject_family_count"],
                )
                self.assertLess(
                    snapshot.counts["modified_mobject_count"],
                    snapshot.counts["mobject_family_count"],
                )
                controller.restore()

    def test_single_controller_publishes_complete_json_safe_frame(self) -> None:
        with patch.dict(os.environ, {QUADRIC_PERFORMANCE_TRACE_ENV: "1"}):
            with tempconfig({"renderer": "cairo"}):
                controller = _single_controller(Scene()).attach()
                snapshot = controller.performance_snapshot()
                self.assertIsNotNone(snapshot)
                assert snapshot is not None
                self.assertEqual(snapshot.schema, QUADRIC_PERFORMANCE_TRACE_SCHEMA)
                self.assertEqual(snapshot.status, "committed")
                required = {
                    "resolve_inputs",
                    "surface_proxy_global_frame",
                    "section_compositing",
                    "contour_union",
                    "boundary_visibility",
                    "boundary_section_spans",
                    "curve_crossings",
                    "boundary_painter_graph",
                    "adaptive_projection",
                    "dash_generation",
                    "painter_band_preparation",
                    "transaction_snapshot",
                    "manim_apply",
                }
                self.assertTrue(required <= set(snapshot.stage_durations_ns))
                self.assertGreater(snapshot.counts["mobject_family_count"], 0)
                self.assertGreater(snapshot.counts["active_mobject_count"], 0)
                self.assertGreater(snapshot.counts["plane_fragment_count"], 0)
                self.assertGreater(snapshot.counts["ray_classification_count"], 0)
                json.dumps(snapshot.to_dict(), sort_keys=True)
                controller.restore()

    def test_apply_failure_records_structured_rollback_evidence(self) -> None:
        with patch.dict(os.environ, {QUADRIC_PERFORMANCE_TRACE_ENV: "true"}):
            with tempconfig({"renderer": "cairo"}):
                controller = _single_controller(Scene()).attach()
                previous = controller.last_frame
                prepared = controller.prepare()
                shifted_items = (
                    replace(
                        prepared.painter_band.items[0],
                        z_index=prepared.painter_band.items[0].z_index + 0.125,
                    ),
                    *prepared.painter_band.items[1:],
                )
                prepared = replace(
                    prepared,
                    painter_band=replace(
                        prepared.painter_band,
                        items=shifted_items,
                    ),
                )
                with patch.object(
                    controller._band,
                    "apply",
                    side_effect=RuntimeError("synthetic apply failure"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "synthetic"):
                        controller.apply(prepared)
                snapshot = controller.performance_snapshot()
                self.assertIsNotNone(snapshot)
                assert snapshot is not None
                self.assertEqual(snapshot.status, "failed")
                self.assertTrue(snapshot.rollback_performed)
                self.assertEqual(snapshot.error_type, "RuntimeError")
                self.assertIn("transaction_rollback", snapshot.stage_durations_ns)
                self.assertIs(controller.last_frame, previous)
                controller.restore()

    def test_composite_controller_uses_the_same_trace_contract(self) -> None:
        cone = ConeSpec(
            "perf-double",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (-2.0, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.OPEN_DOUBLE,
        )
        plane = SectionPlane(
            "perf-double-plane",
            (0.0, 0.5, 0.0),
            (0.0, 1.0, 0.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        with patch.dict(os.environ, {QUADRIC_PERFORMANCE_TRACE_ENV: "yes"}):
            with tempconfig({"renderer": "cairo"}):
                controller = CompositeQuadricSection3D(
                    Scene(),
                    surface=cone,
                    section_id="perf-double-section",
                    plane=plane,
                    projection=SIDE_VIEW,
                    limits=_limits(max_total_mobjects=20000),
                    max_chord_error=0.01,
                ).attach()
                snapshot = controller.performance_snapshot()
                self.assertIsNotNone(snapshot)
                assert snapshot is not None
                self.assertEqual(
                    snapshot.controller_kind,
                    "composite_quadric_section_3d",
                )
                self.assertEqual(snapshot.counts["surface_count"], 2)
                self.assertGreater(snapshot.counts["plane_fragment_count"], 0)
                self.assertIn("contour_union", snapshot.stage_durations_ns)
                controller.update()
                repeated = controller.performance_snapshot()
                self.assertIsNotNone(repeated)
                assert repeated is not None
                self.assertEqual(repeated.counts["display_changed_slot_count"], 0)
                self.assertEqual(
                    repeated.counts["transaction_snapshot_mobject_count"],
                    0,
                )
                self.assertEqual(repeated.counts["modified_mobject_count"], 0)
                controller.restore()


if __name__ == "__main__":
    unittest.main()
