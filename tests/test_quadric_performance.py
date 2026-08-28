from __future__ import annotations

from dataclasses import replace
import json
from math import pi
import os
import unittest
from unittest.mock import patch

from manim import Line, Scene, tempconfig

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
    QuadricManimError,
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
                with patch(
                    "polyhedron_visibility.quadrics.manim."
                    "compute_quadric_section_compositing",
                    side_effect=AssertionError("clean frame recomputed geometry"),
                ):
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
                self.assertEqual(snapshot.cache_hits["dirty_frame"], 1)
                self.assertEqual(snapshot.cache_hits["prepared_numeric"], 1)
                self.assertIn("dirty_frame_shortcut", snapshot.stage_durations_ns)
                self.assertNotIn("section_compositing", snapshot.stage_durations_ns)
                self.assertNotIn("boundary_visibility", snapshot.stage_durations_ns)
                controller.restore()

    def test_dynamic_inputs_are_resolved_once_per_update(self) -> None:
        with tempconfig({"renderer": "cairo"}):
            controller = _single_controller(Scene()).attach()
            names = (
                "_resolve_surfaces",
                "_resolve_curves",
                "_resolve_curve_opacities",
                "_resolve_view",
                "_resolve_section_plane",
                "_resolve_section_patch",
            )
            wrappers = {
                name: patch.object(
                    controller,
                    name,
                    wraps=getattr(controller, name),
                )
                for name in names
            }
            mocks = {name: wrapper.start() for name, wrapper in wrappers.items()}
            try:
                controller.update()
            finally:
                for wrapper in wrappers.values():
                    wrapper.stop()
            self.assertTrue(all(mock.call_count == 1 for mock in mocks.values()))
            controller.restore()

    def test_root_opacity_reuses_numeric_frame_but_updates_display(self) -> None:
        with patch.dict(os.environ, {QUADRIC_PERFORMANCE_TRACE_ENV: "1"}):
            with tempconfig({"renderer": "cairo"}):
                controller = _single_controller(Scene()).attach()
                controller.update()
                controller.display_mobject.set_opacity(0.3)
                with patch(
                    "polyhedron_visibility.quadrics.manim."
                    "compute_quadric_section_compositing",
                    side_effect=AssertionError("opacity frame recomputed geometry"),
                ):
                    controller.update()
                snapshot = controller.performance_snapshot()
                assert snapshot is not None
                self.assertEqual(snapshot.cache_hits["prepared_numeric"], 1)
                self.assertGreater(snapshot.counts["display_changed_slot_count"], 0)
                self.assertGreater(snapshot.counts["modified_mobject_count"], 0)
                self.assertNotIn("section_compositing", snapshot.stage_durations_ns)
                self.assertNotIn("dirty_frame_shortcut", snapshot.stage_durations_ns)
                controller.restore()

    def test_curve_opacity_reuses_geometry_and_changes_active_slots(self) -> None:
        state = {"opacity": 1.0}
        with patch.dict(os.environ, {QUADRIC_PERFORMANCE_TRACE_ENV: "1"}):
            with tempconfig({"renderer": "cairo"}):
                controller = _single_controller(Scene()).attach()
                active_ids = tuple(
                    item.curve_id for item in controller._resolve_curves()
                )
                controller._curve_opacity_input = lambda: {
                    curve_id: state["opacity"] for curve_id in active_ids
                }
                controller.update()
                state["opacity"] = 0.2
                with patch(
                    "polyhedron_visibility.quadrics.manim."
                    "compute_quadric_section_compositing",
                    side_effect=AssertionError("draw-only frame recomputed geometry"),
                ):
                    controller.update()
                snapshot = controller.performance_snapshot()
                assert snapshot is not None
                self.assertEqual(snapshot.cache_hits["prepared_numeric"], 1)
                self.assertGreater(snapshot.counts["display_changed_slot_count"], 0)
                self.assertNotIn("section_compositing", snapshot.stage_durations_ns)
                controller.restore()

    def test_cached_frame_still_rejects_a_painter_band_intruder(self) -> None:
        with patch.dict(os.environ, {QUADRIC_PERFORMANCE_TRACE_ENV: "1"}):
            with tempconfig({"renderer": "cairo"}):
                scene = Scene()
                controller = _single_controller(scene).attach()
                controller.update()
                committed = controller.last_frame
                cached = controller._last_prepared_frame
                intruder = Line((-2, 2, 0), (2, 2, 0)).set_z_index(25.0)
                scene.add(intruder)
                with self.assertRaisesRegex(
                    QuadricManimError,
                    "managed painter z band",
                ):
                    controller.update()
                self.assertIs(controller.last_frame, committed)
                self.assertIs(controller._last_prepared_frame, cached)
                failed = controller.performance_snapshot()
                assert failed is not None
                self.assertEqual(failed.status, "failed")
                scene.remove(intruder)
                controller.update()
                recovered = controller.performance_snapshot()
                assert recovered is not None
                self.assertEqual(recovered.cache_hits["dirty_frame"], 1)
                self.assertGreater(recovered.counts["surface_count"], 0)
                self.assertIn(
                    "dirty_frame_shortcut",
                    recovered.stage_durations_ns,
                )
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
                controller.update()
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
                self.assertEqual(snapshot.cache_misses["dirty_frame"], 1)
                controller.restore()

    def test_failed_full_recompute_keeps_the_last_clean_cache(self) -> None:
        state = {"x": 0.0}

        def surfaces() -> tuple[SphereSpec, ...]:
            return (SphereSpec("cached-sphere", (state["x"], 0.0, 0.0), 1.0),)

        with patch.dict(os.environ, {QUADRIC_PERFORMANCE_TRACE_ENV: "1"}):
            with tempconfig({"renderer": "cairo"}):
                controller = QuadricOcclusion3D(
                    Scene(),
                    surfaces=surfaces,
                    curves=(),
                    projection=VIEW,
                    limits=_limits(),
                ).attach()
                controller.update()
                committed = controller.last_frame
                cached = controller._last_prepared_frame
                state["x"] = 0.25
                with patch(
                    "polyhedron_visibility.quadrics.manim."
                    "compute_global_quadric_frame",
                    side_effect=RuntimeError("synthetic geometry failure"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "synthetic geometry"):
                        controller.update()
                self.assertIs(controller.last_frame, committed)
                self.assertIs(controller._last_prepared_frame, cached)
                state["x"] = 0.0
                controller.update()
                recovered = controller.performance_snapshot()
                assert recovered is not None
                self.assertEqual(recovered.cache_hits["dirty_frame"], 1)
                self.assertIn(
                    "dirty_frame_shortcut",
                    recovered.stage_durations_ns,
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
                self.assertEqual(repeated.cache_hits["dirty_frame"], 1)
                self.assertIn(
                    "dirty_frame_shortcut",
                    repeated.stage_durations_ns,
                )
                self.assertNotIn(
                    "section_compositing",
                    repeated.stage_durations_ns,
                )
                controller.display_mobject.set_opacity(0.4)
                with patch.object(
                    controller,
                    "_local_frames",
                    side_effect=AssertionError(
                        "composite opacity frame recomputed geometry"
                    ),
                ):
                    controller.update()
                faded = controller.performance_snapshot()
                assert faded is not None
                self.assertEqual(faded.cache_hits["prepared_numeric"], 1)
                self.assertGreater(
                    faded.counts["display_changed_slot_count"],
                    0,
                )
                self.assertNotIn(
                    "section_compositing",
                    faded.stage_durations_ns,
                )
                controller.restore()


if __name__ == "__main__":
    unittest.main()
