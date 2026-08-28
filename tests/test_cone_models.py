from __future__ import annotations

from math import cos, pi, sin
import unittest
from unittest.mock import patch

import numpy as np
from manim import Mobject, Scene

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.boundary_compositing import (
    BoundarySourceKind,
    compute_boundary_visibility,
)
from polyhedron_visibility.quadrics.compositing import (
    compute_quadric_compositing,
)
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    PlaneDisplayPatchSpec,
    QuadricContractError,
    SectionPlane,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricManimLimits,
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.plane_patch import fit_plane_display_patch
from polyhedron_visibility.quadrics.global_occlusion import (
    GlobalQuadricOcclusionError,
    compute_global_quadric_frame,
)
from polyhedron_visibility.quadrics.projection import (
    build_cone_projection_layers,
    build_opaque_projection_proxy,
)
from polyhedron_visibility.quadrics.section_compositing import (
    PlaneDepthRole,
    compute_quadric_section_compositing,
    quadric_plane_fragment_contours,
)
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section_boundary_curves,
)
from polyhedron_visibility.quadrics.surface_boundaries import (
    build_surface_boundary_sources,
)
from polyhedron_visibility.quadrics.visibility import (
    compute_quadric_visibility,
)


IDENTITY_VIEW = ParallelView.from_matrix(np.eye(3))
SIDE_VIEW = ParallelView.from_matrix(
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
)
OBLIQUE_VIEW = ParallelView.from_matrix(
    (
        (-0.7071067811865476, 0.7071067811865476, 0.0),
        (-0.4082482904638631, -0.4082482904638631, 0.8164965809277261),
        (0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
    )
)


def _near_side_view(angle: float) -> ParallelView:
    return ParallelView.from_matrix(
        (
            (1.0, 0.0, 0.0),
            (0.0, -sin(angle), cos(angle)),
            (0.0, -cos(angle), -sin(angle)),
        )
    )


def _cone(model: ConeModel, axial_range: tuple[float, float]) -> ConeSpec:
    return ConeSpec(
        "cone",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        pi / 4.0,
        axial_range,
        radial_axis=(1.0, 0.0, 0.0),
        model=model,
    )


class ConeModelContractTests(unittest.TestCase):
    def test_closed_and_open_single_cones_separate_caps_from_trim_rims(self) -> None:
        closed = _cone(ConeModel.CLOSED_SINGLE, (0.0, 2.0))
        opened = _cone(ConeModel.OPEN_SINGLE, (0.0, 2.0))

        self.assertEqual(len(closed.end_caps), 1)
        self.assertEqual(closed.trim_rims, ())
        self.assertEqual(opened.end_caps, ())
        self.assertEqual(len(opened.trim_rims), 1)
        self.assertEqual(opened.trim_rims[0].role, "trim_max")
        self.assertEqual(
            [item.role for item in opened.ray_hits((0, 0, 3), (0, 0, -1))],
            ["support"],
        )
        with self.assertRaisesRegex(QuadricContractError, "no filled-volume"):
            opened.contains((0.0, 0.0, 1.0))

    def test_open_double_expands_to_stable_single_nappe_shells(self) -> None:
        double = _cone(ConeModel.OPEN_DOUBLE, (-3.0, 2.0))
        components = double.render_components
        self.assertEqual(
            tuple(item.surface_id for item in components),
            ("cone:nappe:negative", "cone:nappe:positive"),
        )
        self.assertEqual(
            tuple(item.axial_range for item in components),
            ((-3.0, 0.0), (0.0, 2.0)),
        )
        self.assertTrue(all(item.model is ConeModel.OPEN_SINGLE for item in components))
        self.assertTrue(
            all(item.component_parent_id == "cone" for item in components)
        )
        self.assertEqual(tuple(len(item.trim_rims) for item in components), (1, 1))

    def test_model_and_axial_range_mismatches_fail_explicitly(self) -> None:
        with self.assertRaisesRegex(QuadricContractError, "requires one nappe"):
            _cone(ConeModel.OPEN_SINGLE, (-1.0, 1.0))
        with self.assertRaisesRegex(QuadricContractError, "requires axial_range"):
            _cone(ConeModel.OPEN_DOUBLE, (0.0, 1.0))
        with self.assertRaisesRegex(QuadricContractError, "reserved"):
            ConeSpec(
                "invalid-parent",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                pi / 4.0,
                (0.0, 2.0),
                model=ConeModel.CLOSED_SINGLE,
                component_parent_id="not-a-double",
            )


class ConeProjectionAndSectionTests(unittest.TestCase):
    def test_open_mouth_removes_one_projection_sheet_but_keeps_the_other(self) -> None:
        closed = build_cone_projection_layers(
            _cone(ConeModel.CLOSED_SINGLE, (0.0, 2.0)),
            IDENTITY_VIEW,
            max_chord_error=0.01,
            max_segments=512,
        )
        opened = build_cone_projection_layers(
            _cone(ConeModel.OPEN_SINGLE, (0.0, 2.0)),
            IDENTITY_VIEW,
            max_chord_error=0.01,
            max_segments=512,
        )

        self.assertTrue(closed.terminal_front_facing)
        self.assertEqual(len(closed.front.lateral_paths), 2)
        self.assertEqual(len(closed.front.cap_paths), 1)
        self.assertEqual(len(opened.front.lateral_paths), 2)
        self.assertEqual(opened.front.cap_paths, ())
        self.assertEqual(len(opened.back.lateral_paths), 1)

    def test_component_shading_partitions_both_frustum_terminals(self) -> None:
        for model, terminal_kind in (
            (ConeModel.CLOSED_SINGLE, "cap"),
            (ConeModel.OPEN_SINGLE, "trim"),
        ):
            with self.subTest(model=model.value):
                layers = build_cone_projection_layers(
                    _cone(model, (1.0, 2.0)),
                    IDENTITY_VIEW,
                    max_chord_error=0.01,
                    max_segments=512,
                )
                self.assertIsNone(layers.terminal_front_facing)
                self.assertEqual(
                    dict(layers.terminal_front_facing_by_id),
                    {
                        f"cone:{terminal_kind}:min": False,
                        f"cone:{terminal_kind}:max": True,
                    },
                )
                self.assertEqual(len(layers.back.lateral_paths), 2)
                self.assertEqual(len(layers.front.lateral_paths), 2)
                if model is ConeModel.CLOSED_SINGLE:
                    self.assertEqual(len(layers.back.cap_paths), 1)
                    self.assertEqual(len(layers.front.cap_paths), 1)
                    self.assertEqual(len(layers.opaque_lateral_paths), 2)
                    self.assertEqual(len(layers.opaque_cap_paths), 1)
                else:
                    self.assertEqual(layers.back.cap_paths, ())
                    self.assertEqual(layers.front.cap_paths, ())
                    self.assertEqual(len(layers.opaque_lateral_paths), 1)
                    self.assertEqual(layers.opaque_cap_paths, ())

    def test_edge_on_frustum_terminals_do_not_invent_fill_area(self) -> None:
        for model in (ConeModel.CLOSED_SINGLE, ConeModel.OPEN_SINGLE):
            with self.subTest(model=model.value):
                layers = build_cone_projection_layers(
                    _cone(model, (1.0, 2.0)),
                    SIDE_VIEW,
                    max_chord_error=0.01,
                    max_segments=512,
                )
                self.assertEqual(
                    set(dict(layers.terminal_front_facing_by_id).values()),
                    {None},
                )
                self.assertEqual(len(layers.back.lateral_paths), 1)
                self.assertEqual(len(layers.front.lateral_paths), 1)
                self.assertEqual(layers.back.cap_paths, ())
                self.assertEqual(layers.front.cap_paths, ())

    def test_open_shell_section_roles_do_not_use_a_missing_cap(self) -> None:
        plane = SectionPlane(
            "cut",
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        patch = PlaneDisplayPatchSpec("patch", plane.plane_id, 2.5, 2.5)
        roles_by_model: dict[ConeModel, set[PlaneDepthRole]] = {}
        for model in (ConeModel.CLOSED_SINGLE, ConeModel.OPEN_SINGLE):
            cone = _cone(model, (0.0, 2.0))
            proxy = build_opaque_projection_proxy(
                cone,
                IDENTITY_VIEW,
                max_chord_error=0.08,
            )
            base = compute_quadric_compositing(
                compute_quadric_visibility((), (cone,), IDENTITY_VIEW),
                (proxy,),
            )
            frame = compute_quadric_section_compositing(
                base,
                cone,
                plane,
                patch,
                IDENTITY_VIEW,
                max_screen_error=0.2,
            )
            roles_by_model[model] = {item.role for item in frame.plane_fragments}

        self.assertIn(
            PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
            roles_by_model[ConeModel.CLOSED_SINGLE],
        )
        self.assertNotIn(
            PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
            roles_by_model[ConeModel.OPEN_SINGLE],
        )
        self.assertIn(
            PlaneDepthRole.IN_FRONT_OF_SURFACE,
            roles_by_model[ConeModel.OPEN_SINGLE],
        )

    def test_open_shell_plane_roles_follow_trim_rim_not_a_false_chord(self) -> None:
        cone = ConeSpec(
            "open-shell-trim-regression",
            (0.0, 0.0, -2.4),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 4.0),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.OPEN_SINGLE,
        )
        normal = np.asarray((0.82, 0.0, 1.0), dtype=float)
        normal /= np.linalg.norm(normal)
        plane = SectionPlane(
            "open-shell-trim-cut",
            tuple(np.asarray((0.0, 0.0, -0.35)) + 0.48 * normal),
            (0.82, 0.0, 1.0),
            u_axis=(0.0, 1.0, 0.0),
        )
        patch = fit_plane_display_patch(
            "open-shell-trim-patch",
            plane,
            (cone,),
            margin_ratio=0.08,
        ).patch
        proxy = build_opaque_projection_proxy(
            cone,
            OBLIQUE_VIEW,
            max_chord_error=0.008,
            max_segments=768,
        )
        base = compute_quadric_compositing(
            compute_quadric_visibility((), (cone,), OBLIQUE_VIEW),
            (proxy,),
        )
        frame = compute_quadric_section_compositing(
            base,
            cone,
            plane,
            patch,
            OBLIQUE_VIEW,
            max_screen_error=0.08,
        )
        contours = quadric_plane_fragment_contours(frame)

        def contour_edges(role: PlaneDepthRole):
            result = {}
            for contour in contours[role]:
                points = tuple(tuple(float(value) for value in point) for point in contour)
                for start, end in zip(points, (*points[1:], points[0])):
                    key = tuple(sorted((start, end)))
                    result[key] = float(
                        np.linalg.norm(np.asarray(end) - np.asarray(start))
                    )
            return result

        between_edges = contour_edges(PlaneDepthRole.BETWEEN_SURFACE_SHEETS)
        front_edges = contour_edges(PlaneDepthRole.IN_FRONT_OF_SURFACE)
        shared_lengths = tuple(
            between_edges[key] for key in between_edges.keys() & front_edges.keys()
        )
        self.assertTrue(shared_lengths)

        # The former regression emitted one 3.65-unit straight chord through
        # the open mouth.  The real boundary is now the sampled trim-rim arc;
        # every shared segment remains one local piece of that arc.
        self.assertLess(max(shared_lengths), 0.32)
        self.assertLess(len(frame.plane_fragments), 8192)
        self.assertLess(frame.ray_classification_count, 65536)

    def test_side_view_open_cone_and_frustum_use_proxy_owned_rim_segments(
        self,
    ) -> None:
        plane = SectionPlane(
            "side-view-cut",
            (0.0, 0.5, 0.0),
            (0.0, 1.0, 0.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        patch = PlaneDisplayPatchSpec(
            "side-view-patch",
            plane.plane_id,
            3.0,
            3.0,
        )
        for axial_range, expected_proxy_vertices, expected_rims in (
            ((0.0, 2.0), 3, 1),
            ((1.0, 2.0), 4, 2),
        ):
            with self.subTest(axial_range=axial_range):
                cone = _cone(ConeModel.OPEN_SINGLE, axial_range)
                proxy = build_opaque_projection_proxy(
                    cone,
                    SIDE_VIEW,
                    max_chord_error=0.008,
                    max_segments=768,
                )
                base = compute_quadric_compositing(
                    compute_quadric_visibility((), (cone,), SIDE_VIEW),
                    (proxy,),
                )
                frame = compute_quadric_section_compositing(
                    base,
                    cone,
                    plane,
                    patch,
                    SIDE_VIEW,
                    max_screen_error=0.08,
                )

                self.assertEqual(len(cone.trim_rims), expected_rims)
                self.assertEqual(len(proxy.vertices), expected_proxy_vertices)
                self.assertTrue(frame.plane_fragments)
                self.assertLess(len(frame.plane_fragments), 8192)
                self.assertLess(frame.ray_classification_count, 65536)
                self.assertIn(
                    PlaneDepthRole.IN_FRONT_OF_SURFACE,
                    {fragment.role for fragment in frame.plane_fragments},
                )


class ConeBoundaryAndBindingTests(unittest.TestCase):
    def test_open_trim_rim_has_owner_aware_solid_and_hidden_halves(self) -> None:
        cone = _cone(ConeModel.OPEN_SINGLE, (0.0, 2.0))
        sources = build_surface_boundary_sources((cone,), SIDE_VIEW)
        rim = next(item for item in sources if "trim_max" in item.source_id)
        self.assertIs(rim.source_kind, BoundarySourceKind.SURFACE_TRIM_RIM)
        spans = compute_boundary_visibility((rim,), (cone,), SIDE_VIEW)[rim.source_id]
        self.assertEqual(
            tuple(item.kind.value for item in spans),
            ("visible", "hidden"),
        )
        self.assertEqual(spans[1].occluder_surface_ids, ("cone",))

    def test_double_shell_binding_expands_once_and_updates_without_allocation(
        self,
    ) -> None:
        double = _cone(ConeModel.OPEN_DOUBLE, (-2.0, 2.0))
        controller = QuadricOcclusion3D(
            Scene(),
            surfaces=(double,),
            curves=(),
            projection=SIDE_VIEW,
            boundary_visibility_mode="unified",
            style=QuadricManimStyle(
                cone_lateral_fill_colors=("#173753", "#4F84B3", "#1D4368"),
            ),
        ).attach()
        self.assertEqual(
            controller._surface_ids,
            ("cone:nappe:negative", "cone:nappe:positive"),
        )
        frame = controller.last_global_frame
        assert frame is not None
        self.assertEqual(
            frame.surface_depth_evidence[0].projection_relation,
            "touching_open_double_nappes",
        )
        identities = controller.slot_identities()
        controller.update()
        self.assertEqual(controller.slot_identities(), identities)
        controller.restore()

    def test_near_side_rim_rank_switch_preserves_slots_and_commits_every_frame(
        self,
    ) -> None:
        state = {"angle": 0.015}

        def current_view(scene: object) -> ParallelView:
            del scene
            return _near_side_view(state["angle"])

        cone = _cone(ConeModel.OPEN_SINGLE, (0.0, 2.0))
        plane = SectionPlane(
            "animated-side-cut",
            (0.0, 0.5, 0.0),
            (0.0, 1.0, 0.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        patch_spec = PlaneDisplayPatchSpec(
            "animated-side-patch",
            plane.plane_id,
            3.0,
            3.0,
        )
        section_curves = compute_quadric_section_boundary_curves(
            "animated-side-section",
            cone,
            plane,
        )
        controller = QuadricOcclusion3D(
            Scene(),
            surfaces=(cone,),
            curves=section_curves,
            projection=current_view,
            section_plane=plane,
            section_patch=patch_spec,
            boundary_visibility_mode="unified",
            include_surface_boundaries=True,
            max_chord_error=0.008,
            section_max_screen_error=0.08,
            limits=QuadricManimLimits(
                max_surfaces=1,
                max_curves=1,
                max_fragments_per_curve=16,
                max_segments_per_fragment=96,
                max_surface_segments=768,
                max_dashes_per_fragment=40,
                max_projected_length=12.0,
                max_total_mobjects=10000,
                max_boundary_sources=16,
            ),
            style=QuadricManimStyle(
                cone_lateral_fill_colors=None,
                cone_cap_fill_colors=None,
            ),
        ).attach()
        try:
            identities = controller.slot_identities()
            previous_frame = controller.last_section_frame
            committed_counts = []
            with patch.object(
                Mobject,
                "__init__",
                side_effect=AssertionError("rank switch allocated a Mobject"),
            ):
                for angle in (
                    0.005,
                    0.0026,
                    0.0025,
                    0.0024,
                    0.0,
                    -0.0024,
                    -0.0026,
                    -0.005,
                    -0.015,
                ):
                    state["angle"] = angle
                    controller.update()
                    frame = controller.last_section_frame
                    self.assertIsNotNone(frame)
                    self.assertIsNot(frame, previous_frame)
                    self.assertEqual(controller.slot_identities(), identities)
                    committed_counts.append(len(frame.plane_fragments))
                    previous_frame = frame
            self.assertGreater(len(set(committed_counts)), 1)
        finally:
            controller.restore()

    def test_unrelated_open_shells_cannot_claim_the_double_apex_exception(self) -> None:
        impostors = (
            ConeSpec(
                "impostor-negative",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                pi / 4.0,
                (-2.0, 0.0),
                radial_axis=(1.0, 0.0, 0.0),
                model=ConeModel.OPEN_SINGLE,
                component_parent_id="claimed-double",
            ),
            ConeSpec(
                "impostor-positive",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                pi / 4.0,
                (0.0, 2.0),
                radial_axis=(1.0, 0.0, 0.0),
                model=ConeModel.OPEN_SINGLE,
                component_parent_id="claimed-double",
            ),
        )
        with self.assertRaisesRegex(
            GlobalQuadricOcclusionError,
            "touching, intersecting, or numerically inseparable",
        ):
            compute_global_quadric_frame((), impostors, SIDE_VIEW)


if __name__ == "__main__":
    unittest.main()
