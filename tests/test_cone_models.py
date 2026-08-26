from __future__ import annotations

from math import pi
import unittest

import numpy as np
from manim import Scene

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
    QuadricManimStyle,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.global_occlusion import (
    GlobalQuadricOcclusionError,
    compute_global_quadric_frame,
)
from polyhedron_visibility.quadrics.projection import (
    ProjectionProxyError,
    build_cone_projection_layers,
    build_opaque_projection_proxy,
)
from polyhedron_visibility.quadrics.section_compositing import (
    PlaneDepthRole,
    compute_quadric_section_compositing,
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

    def test_component_shading_rejects_two_terminal_frustum_masks(self) -> None:
        with self.assertRaisesRegex(ProjectionProxyError, "apex-to-one-rim"):
            build_cone_projection_layers(
                _cone(ConeModel.CLOSED_SINGLE, (1.0, 2.0)),
                IDENTITY_VIEW,
            )

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
