from __future__ import annotations

from dataclasses import replace
from math import pi, sqrt, tau
import unittest
from unittest.mock import patch

import numpy as np

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.boundary_compositing import (
    BoundaryOcclusionScope,
    BoundarySemanticKind,
    BoundarySourceKind,
)
from polyhedron_visibility.quadrics.contract import (
    CircularTrimRimSpec,
    ConeModel,
    ConeSpec,
    CylinderSpec,
    PlanarCapSpec,
)
from polyhedron_visibility.quadrics.curves import CircleArcCurve
from polyhedron_visibility.quadrics.planar_curves import (
    Circle3DSpec,
    PlanarCurve3DContractError,
    PlanarFrame3D,
)
from polyhedron_visibility.quadrics.surface_boundaries import (
    build_surface_boundary_sources,
)
from polyhedron_visibility.topology import ParameterInterval


VIEW = ParallelView.from_matrix(
    (
        (-1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0),
        (-1.0 / sqrt(6.0), -1.0 / sqrt(6.0), 2.0 / sqrt(6.0)),
        (1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0)),
    )
)


class PlanarSurfaceRimContractTests(unittest.TestCase):
    def assert_vector_close(
        self,
        actual: object,
        expected: object,
        *,
        atol: float = 1.0e-12,
    ) -> None:
        np.testing.assert_allclose(
            np.asarray(actual, dtype=float),
            np.asarray(expected, dtype=float),
            rtol=0.0,
            atol=atol,
        )

    def assert_support_circle(
        self,
        boundary: PlanarCapSpec | CircularTrimRimSpec,
        curve_id: str,
    ) -> Circle3DSpec:
        frame = boundary.planar_frame
        self.assertIsInstance(frame, PlanarFrame3D)
        self.assertEqual(
            frame.frame_id,
            boundary.cap_id
            if isinstance(boundary, PlanarCapSpec)
            else boundary.rim_id,
        )
        self.assert_vector_close(frame.point, boundary.center)
        self.assert_vector_close(frame.normal, boundary.normal)
        self.assert_vector_close(frame.u_axis, boundary.radial_axis)

        circle = boundary.boundary_circle(curve_id)
        self.assertIsInstance(circle, Circle3DSpec)
        self.assertEqual(circle.curve_id, curve_id)
        self.assertEqual(circle.frame, frame)
        self.assert_vector_close(circle.center, boundary.center)
        self.assertEqual(circle.radius, boundary.radius)
        self.assertEqual(circle.domain, ParameterInterval(0.0, tau))

        lowered = circle.lower_to_analytic_curve()
        self.assertIsInstance(lowered, CircleArcCurve)
        self.assertEqual(lowered.curve_id, curve_id)
        self.assertEqual(lowered.domain, ParameterInterval(0.0, tau))
        self.assertTrue(lowered.closed)
        self.assert_vector_close(lowered.center, boundary.center)
        self.assert_vector_close(lowered.normal, boundary.normal)

        radial = np.asarray(boundary.radial_axis, dtype=float)
        normal = np.asarray(boundary.normal, dtype=float)
        second = np.cross(normal, radial)
        self.assert_vector_close(
            lowered.first_axis,
            boundary.radius * radial,
        )
        self.assert_vector_close(
            lowered.second_axis,
            boundary.radius * second,
        )
        self.assert_vector_close(
            lowered.point(0.0),
            np.asarray(boundary.center) + boundary.radius * radial,
        )
        self.assert_vector_close(
            lowered.tangent(0.0),
            boundary.radius * second,
        )
        return circle

    def assert_source_contract(
        self,
        source: object,
        boundary: PlanarCapSpec | CircularTrimRimSpec,
        *,
        source_kind: BoundarySourceKind,
    ) -> None:
        source_id = (
            f"boundary:{boundary.parent_surface_id}:{boundary.role}:rim"
        )
        owner_id = (
            boundary.cap_id
            if isinstance(boundary, PlanarCapSpec)
            else boundary.rim_id
        )
        self.assertEqual(source.source_id, source_id)
        self.assertIs(source.source_kind, source_kind)
        self.assertIs(
            source.semantic_kind,
            BoundarySemanticKind.SURFACE_BOUNDARY,
        )
        self.assertIs(
            source.occlusion_scope,
            BoundaryOcclusionScope.OWNER_AND_EXTERNAL,
        )
        self.assertEqual(source.owner_id, owner_id)
        self.assertEqual(
            source.owner_surface_id,
            boundary.parent_surface_id,
        )
        self.assertEqual(source.style_id, "style:surface-boundary")
        self.assertEqual(
            source.stable_sort_key,
            (
                source_kind.value,
                BoundarySemanticKind.SURFACE_BOUNDARY.value,
                source_id,
            ),
        )
        self.assertEqual(source.curve.curve_id, source_id)
        self.assertEqual(source.curve.domain, ParameterInterval(0.0, tau))
        self.assertTrue(source.curve.closed)

        expected = boundary.boundary_circle(
            source_id
        ).lower_to_analytic_curve()
        for parameter in (0.0, 0.25 * pi, 0.5 * pi, pi, 1.5 * pi, tau):
            self.assert_vector_close(
                source.curve.point(parameter),
                expected.point(parameter),
            )
            self.assert_vector_close(
                source.curve.tangent(parameter),
                expected.tangent(parameter),
            )

    def test_planar_cap_exposes_stable_frame_and_circle(self) -> None:
        cap = PlanarCapSpec(
            "cap:slanted",
            "surface:slanted",
            (1.25, -0.5, 2.0),
            (1.0, 2.0, 3.0),
            1.75,
            radial_axis=(2.0, -1.0, 0.0),
            role="cap_max",
        )
        self.assert_support_circle(cap, "boundary:cap:slanted")

    def test_trim_rim_exposes_stable_frame_and_circle(self) -> None:
        rim = CircularTrimRimSpec(
            "rim:slanted",
            "surface:slanted",
            (-0.75, 1.5, 0.25),
            (-1.0, -2.0, -3.0),
            0.875,
            radial_axis=(2.0, -1.0, 0.0),
            role="trim_min",
        )
        self.assert_support_circle(rim, "boundary:rim:slanted")

    def test_boundary_replace_preserves_certified_frame_and_payload(self) -> None:
        common = dict(
            parent_surface_id="surface:replace",
            center=(
                0.2928276418206841,
                -0.9698281416274352,
                -0.30029473129904827,
            ),
            normal=(
                1.4132667908068457,
                0.03110221354462144,
                -0.04297320205816883,
            ),
            radius=1.0e4,
            radial_axis=(
                -1.7089423051639878,
                -0.29089177871842153,
                -0.42087323476286687,
            ),
        )
        boundaries = (
            PlanarCapSpec("cap:replace", role="cap_max", **common),
            CircularTrimRimSpec("rim:replace", role="trim_max", **common),
        )

        for boundary in boundaries:
            with self.subTest(boundary=type(boundary).__name__):
                rebuilt = replace(boundary)
                curve_id = f"boundary:{boundary.parent_surface_id}:{boundary.role}:rim"
                self.assertEqual(rebuilt, boundary)
                self.assertEqual(rebuilt.planar_frame, boundary.planar_frame)
                self.assertEqual(
                    rebuilt.planar_frame.to_dict(),
                    boundary.planar_frame.to_dict(),
                )
                self.assertEqual(
                    rebuilt.boundary_circle(curve_id).to_dict(),
                    boundary.boundary_circle(curve_id).to_dict(),
                )

    def test_boundary_replace_reauthors_changed_public_frame_fields(self) -> None:
        boundaries = (
            PlanarCapSpec(
                "cap:old",
                "surface:replace",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                2.0,
                radial_axis=(1.0, 0.0, 0.0),
                role="cap_max",
            ),
            CircularTrimRimSpec(
                "rim:old",
                "surface:replace",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                2.0,
                radial_axis=(1.0, 0.0, 0.0),
                role="trim_max",
            ),
        )

        for boundary in boundaries:
            with self.subTest(boundary=type(boundary).__name__):
                identity_field = (
                    {"cap_id": "cap:new"}
                    if isinstance(boundary, PlanarCapSpec)
                    else {"rim_id": "rim:new"}
                )
                changed = replace(
                    boundary,
                    **identity_field,
                    center=(4.0, 5.0, 6.0),
                    normal=(0.0, 1.0, 0.0),
                    radial_axis=(1.0, 0.0, 0.0),
                    radius=3.0,
                )
                expected_id = (
                    changed.cap_id
                    if isinstance(changed, PlanarCapSpec)
                    else changed.rim_id
                )
                self.assertEqual(changed.planar_frame.frame_id, expected_id)
                self.assertEqual(changed.planar_frame.point, changed.center)
                self.assertEqual(changed.planar_frame.normal, changed.normal)
                self.assertEqual(
                    changed.planar_frame.u_axis,
                    changed.radial_axis,
                )
                self.assertEqual(changed.radius, 3.0)
                self.assert_support_circle(changed, f"boundary:{expected_id}")

    def test_slanted_cylinder_cap_sources_preserve_complete_contract(self) -> None:
        cylinder = CylinderSpec(
            "slanted-cylinder",
            (0.25, -1.0, 2.5),
            (1.0, 2.0, 3.0),
            1.25,
            (-0.75, 2.5),
            radial_axis=(2.0, -1.0, 0.0),
        )
        sources = build_surface_boundary_sources(
            (cylinder,),
            VIEW,
            include_silhouettes=False,
        )
        self.assertEqual(len(sources), 2)
        by_id = {source.source_id: source for source in sources}
        for cap in cylinder.end_caps:
            source_id = f"boundary:{cylinder.surface_id}:{cap.role}:rim"
            self.assertIn(source_id, by_id)
            self.assert_source_contract(
                by_id[source_id],
                cap,
                source_kind=BoundarySourceKind.SURFACE_CAP_RIM,
            )

    def test_closed_and_open_single_cones_use_distinct_real_boundaries(
        self,
    ) -> None:
        common = dict(
            apex=(0.5, -0.25, -1.0),
            axis=(0.0, 0.0, 1.0),
            half_angle=pi / 6.0,
            axial_range=(0.0, 3.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        closed = ConeSpec(
            "closed-cone",
            model=ConeModel.CLOSED_SINGLE,
            **common,
        )
        opened = ConeSpec(
            "open-cone",
            model=ConeModel.OPEN_SINGLE,
            **common,
        )

        closed_sources = build_surface_boundary_sources(
            (closed,),
            VIEW,
            include_silhouettes=False,
        )
        open_sources = build_surface_boundary_sources(
            (opened,),
            VIEW,
            include_silhouettes=False,
        )
        self.assertEqual(len(closed.end_caps), 1)
        self.assertEqual(len(closed.trim_rims), 0)
        self.assertEqual(len(closed_sources), 1)
        self.assert_source_contract(
            closed_sources[0],
            closed.end_caps[0],
            source_kind=BoundarySourceKind.SURFACE_CAP_RIM,
        )
        self.assertEqual(len(opened.end_caps), 0)
        self.assertEqual(len(opened.trim_rims), 1)
        self.assertEqual(len(open_sources), 1)
        self.assert_source_contract(
            open_sources[0],
            opened.trim_rims[0],
            source_kind=BoundarySourceKind.SURFACE_TRIM_RIM,
        )

    def test_frustum_preserves_both_terminal_rims(self) -> None:
        for model, role_names, source_kind in (
            (
                ConeModel.CLOSED_SINGLE,
                ("cap_min", "cap_max"),
                BoundarySourceKind.SURFACE_CAP_RIM,
            ),
            (
                ConeModel.OPEN_SINGLE,
                ("trim_min", "trim_max"),
                BoundarySourceKind.SURFACE_TRIM_RIM,
            ),
        ):
            with self.subTest(model=model.value):
                frustum = ConeSpec(
                    f"{model.value}-frustum",
                    (0.0, 0.0, 0.0),
                    (1.0, 2.0, 3.0),
                    pi / 5.0,
                    (1.0, 4.0),
                    radial_axis=(2.0, -1.0, 0.0),
                    model=model,
                )
                boundaries = (
                    frustum.end_caps
                    if model is ConeModel.CLOSED_SINGLE
                    else frustum.trim_rims
                )
                self.assertEqual(
                    tuple(boundary.role for boundary in boundaries),
                    role_names,
                )
                sources = build_surface_boundary_sources(
                    (frustum,),
                    VIEW,
                    include_silhouettes=False,
                )
                self.assertEqual(len(sources), 2)
                by_id = {source.source_id: source for source in sources}
                for boundary in boundaries:
                    source_id = (
                        f"boundary:{frustum.surface_id}:{boundary.role}:rim"
                    )
                    self.assertIn(source_id, by_id)
                    self.assert_source_contract(
                        by_id[source_id],
                        boundary,
                        source_kind=source_kind,
                    )

    def test_surface_source_builder_delegates_to_unified_circle_contract(
        self,
    ) -> None:
        cylinder = CylinderSpec(
            "delegated-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            (-1.0, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        cone = ConeSpec(
            "delegated-shell",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (0.0, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.OPEN_SINGLE,
        )
        cap_method = PlanarCapSpec.boundary_circle
        rim_method = CircularTrimRimSpec.boundary_circle
        with (
            patch.object(
                PlanarCapSpec,
                "boundary_circle",
                autospec=True,
                side_effect=cap_method,
            ) as cap_calls,
            patch.object(
                CircularTrimRimSpec,
                "boundary_circle",
                autospec=True,
                side_effect=rim_method,
            ) as rim_calls,
        ):
            sources = build_surface_boundary_sources(
                (cylinder, cone),
                VIEW,
                include_silhouettes=False,
            )

        self.assertEqual(len(sources), 3)
        self.assertEqual(cap_calls.call_count, 2)
        self.assertEqual(rim_calls.call_count, 1)
        self.assertEqual(
            {call.args[1] for call in cap_calls.call_args_list},
            {
                "boundary:delegated-cylinder:cap_min:rim",
                "boundary:delegated-cylinder:cap_max:rim",
            },
        )
        self.assertEqual(
            {call.args[1] for call in rim_calls.call_args_list},
            {"boundary:delegated-shell:trim_max:rim"},
        )

    def test_extreme_boundary_scales_fail_closed_during_circle_lowering(
        self,
    ) -> None:
        tiny = PlanarCapSpec(
            "cap:tiny",
            "surface:tiny",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0e-200,
            radial_axis=(1.0, 0.0, 0.0),
        )
        huge = CircularTrimRimSpec(
            "rim:huge",
            "surface:huge",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0e155,
            radial_axis=(1.0, 0.0, 0.0),
        )
        for boundary, curve_id in (
            (tiny, "boundary:tiny"),
            (huge, "boundary:huge"),
        ):
            with self.subTest(curve_id=curve_id):
                with self.assertRaisesRegex(
                    PlanarCurve3DContractError,
                    "certifiable numeric range|cannot be lowered",
                ):
                    boundary.boundary_circle(curve_id)

    def test_surface_builder_does_not_publish_uncertifiable_rim_curve(
        self,
    ) -> None:
        cylinder = CylinderSpec(
            "huge-cylinder",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0e155,
            (0.0, 1.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "certifiable numeric range",
        ):
            build_surface_boundary_sources(
                (cylinder,),
                VIEW,
                include_silhouettes=False,
            )

    def test_large_translation_cannot_swallow_surface_rim_radius(self) -> None:
        rim = CircularTrimRimSpec(
            "rim:translated",
            "surface:translated",
            (1.0e18, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
            radial_axis=(1.0, 0.0, 0.0),
        )

        with self.assertRaisesRegex(
            PlanarCurve3DContractError,
            "semi-axis is not representable",
        ):
            rim.boundary_circle("boundary:translated")


if __name__ == "__main__":
    unittest.main()
