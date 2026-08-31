from __future__ import annotations

import ast
import inspect
from math import pi, sqrt
from pathlib import Path
import unittest

import numpy as np
from manim import Line, VGroup

from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    SectionPlane,
)
from polyhedron_visibility.quadrics.dandelin import (
    DandelinConicFamily,
    compute_dandelin_construction,
)
import tikz_native.dandelin_fixed_view as fixed_view_module
from tikz_native.dandelin_fixed_view import (
    DandelinFixedViewError,
    build_dandelin_fixed_view,
)


HALF_ANGLE = pi / 6.0
SPATIAL_MATRIX = np.asarray(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.8, 0.6),
        (0.0, -0.6, 0.8),
    ),
    dtype=float,
)


def _normal_with_axis_dot(value: float) -> tuple[float, float, float]:
    return (sqrt(max(0.0, 1.0 - value * value)), 0.0, value)


def _cone(
    model: ConeModel,
    axial_range: tuple[float, float],
    *,
    surface_id: str = "cone",
) -> ConeSpec:
    return ConeSpec(
        surface_id,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        HALF_ANGLE,
        axial_range,
        radial_axis=(1.0, 0.0, 0.0),
        model=model,
    )


def _plane(axis_dot: float, *, plane_id: str = "section-plane") -> SectionPlane:
    return SectionPlane(
        plane_id,
        (0.0, 0.0, 2.0),
        _normal_with_axis_dot(axis_dot),
        u_axis=(0.0, 1.0, 0.0),
    )


def _construction(
    construction_id: str = "fixed-view",
    *,
    axis_dot: float = 0.8,
    model: ConeModel = ConeModel.OPEN_SINGLE,
    axial_range: tuple[float, float] = (0.0, 20.0),
):
    return compute_dandelin_construction(
        construction_id,
        _cone(model, axial_range),
        _plane(axis_dot),
    )


def _semantic_objects(group: VGroup, kind: str | None = None) -> tuple[object, ...]:
    values = tuple(
        item
        for item in group.submobjects
        if isinstance(getattr(item, "dandelin_metadata", None), dict)
    )
    if kind is None:
        return values
    return tuple(
        item
        for item in values
        if item.dandelin_metadata.get("semanticKind") == kind
    )


def _semantic_signature(group: VGroup) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            item.dandelin_metadata["semanticKind"],
            item.dandelin_metadata["semanticId"],
            item.dandelin_metadata["semanticSourceRefs"],
            item.dandelin_metadata.get("projectionRank"),
        )
        for item in _semantic_objects(group)
    )


class DandelinFixedViewTests(unittest.TestCase):
    def test_teaching_transparent_mode_uses_certified_surface_painter_items(
        self,
    ) -> None:
        construction = _construction("teaching-transparent")
        group = build_dandelin_fixed_view(
            construction,
            view="spatial",
            projection_matrix=SPATIAL_MATRIX,
            mode="depth_aware_teaching_transparent",
        )

        self.assertTrue(group.curve_visibility_authoritative)
        self.assertTrue(group.surface_layering_authoritative)
        self.assertFalse(group.surface_visibility_authoritative)
        self.assertFalse(group.physical_surface_visibility_authoritative)
        self.assertIsNotNone(group.surface_layer_frame)
        self.assertGreater(group.metadata["planeFragmentCount"], 0)
        self.assertEqual(
            group.metadata["equalDepthContactCount"],
            len(construction.spheres),
        )
        paint_roots = {
            member.dandelin_metadata["paintItemId"]: member
            for wrapper in group.submobjects
            for member in wrapper.submobjects
            if isinstance(getattr(member, "dandelin_metadata", None), dict)
            and "paintItemId" in member.dandelin_metadata
        }
        frame = group.surface_layer_frame
        self.assertTrue(set(frame.draw_order).issubset(paint_roots))
        self.assertEqual(
            tuple(
                item_id
                for item_id in sorted(
                    frame.draw_order,
                    key=lambda value: paint_roots[value].z_index,
                )
            ),
            frame.draw_order,
        )
        rank = {
            item_id: paint_roots[item_id].z_index
            for item_id in frame.draw_order
        }
        cone = frame.cone_layers[0]
        cone_sheet_alpha = tuple(
            paint_roots[item_id].dandelin_metadata["fillOpacity"]
            for item_id in (cone.back_item_id, cone.front_item_id)
        )
        self.assertAlmostEqual(cone_sheet_alpha[0], cone_sheet_alpha[1])
        self.assertAlmostEqual(
            1.0 - (1.0 - cone_sheet_alpha[0]) ** 2,
            0.13,
            places=12,
        )
        for sphere in frame.sphere_layers:
            self.assertLess(rank[cone.back_item_id], rank[sphere.item_id])
            self.assertLess(rank[sphere.item_id], rank[cone.front_item_id])
        plane_items = (*frame.plane_layers, *frame.plane_outline_layers)
        far_sphere = next(
            item
            for item in frame.sphere_layers
            if item.plane_position.value == "in_front_of_sphere"
        )
        near_sphere = next(
            item
            for item in frame.sphere_layers
            if item.plane_position.value == "behind_sphere"
        )
        for item in plane_items:
            if item.role.value == "behind_surface":
                self.assertLess(rank[item.item_id], rank[far_sphere.item_id])
            else:
                self.assertLess(rank[far_sphere.item_id], rank[item.item_id])
            if item.role.value == "in_front_of_surface":
                self.assertLess(rank[near_sphere.item_id], rank[item.item_id])
            else:
                self.assertLess(rank[item.item_id], rank[near_sphere.item_id])
        hidden = tuple(
            item
            for item in paint_roots.values()
            if item.dandelin_metadata.get("renderIntent") == "dashed"
        )
        visible = tuple(
            item
            for item in paint_roots.values()
            if item.dandelin_metadata.get("renderIntent") == "solid"
        )
        self.assertTrue(hidden)
        self.assertTrue(visible)
        self.assertLess(max(rank.values()), min(item.z_index for item in visible))
        contact_fragments = tuple(
            member
            for wrapper in group.submobjects
            if wrapper.dandelin_metadata["semanticKind"] == "contact_circle"
            for member in wrapper.submobjects
        )
        self.assertTrue(contact_fragments)
        self.assertTrue(
            all(
                item.dandelin_metadata["equalDepthFeatureOwner"] is True
                for item in contact_fragments
            )
        )

    def test_teaching_transparent_same_screen_reverse_depth_swaps_spheres(
        self,
    ) -> None:
        construction = _construction("teaching-reverse-depth")
        reverse = SPATIAL_MATRIX.copy()
        reverse[2] *= -1.0
        groups = tuple(
            build_dandelin_fixed_view(
                construction,
                view="spatial",
                projection_matrix=matrix,
                mode="depth_aware_teaching_transparent",
            )
            for matrix in (SPATIAL_MATRIX, reverse)
        )

        first, second = (item.surface_layer_frame for item in groups)
        self.assertEqual(first.projection_matrix[:2], second.projection_matrix[:2])
        self.assertEqual(
            first.sphere_pair_evidence[0].farther_sphere_id,
            second.sphere_pair_evidence[0].nearer_sphere_id,
        )
        self.assertNotEqual(first.draw_order, second.draw_order)
        for group in groups:
            frame = group.surface_layer_frame
            paint_roots = {
                member.dandelin_metadata["paintItemId"]: member
                for wrapper in group.submobjects
                for member in wrapper.submobjects
                if isinstance(getattr(member, "dandelin_metadata", None), dict)
                and "paintItemId" in member.dandelin_metadata
            }
            rank = {
                item_id: paint_roots[item_id].z_index
                for item_id in frame.draw_order
            }
            far_sphere = next(
                item
                for item in frame.sphere_layers
                if item.plane_position.value == "in_front_of_sphere"
            )
            near_sphere = next(
                item
                for item in frame.sphere_layers
                if item.plane_position.value == "behind_sphere"
            )
            for item in (*frame.plane_layers, *frame.plane_outline_layers):
                if item.role.value == "behind_surface":
                    self.assertLess(rank[item.item_id], rank[far_sphere.item_id])
                else:
                    self.assertLess(rank[far_sphere.item_id], rank[item.item_id])
                if item.role.value == "in_front_of_surface":
                    self.assertLess(rank[near_sphere.item_id], rank[item.item_id])
                else:
                    self.assertLess(rank[item.item_id], rank[near_sphere.item_id])

    def test_teaching_transparent_mode_fails_closed_outside_certified_scope(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            DandelinFixedViewError,
            "show_contact_circles=true.*equal-depth seams",
        ):
            build_dandelin_fixed_view(
                _construction("teaching-hidden-equal-depth-seams"),
                view="spatial",
                projection_matrix=SPATIAL_MATRIX,
                mode="depth_aware_teaching_transparent",
                show_contact_circles=False,
            )
        with self.assertRaisesRegex(
            DandelinFixedViewError,
            "only valid for the spatial view",
        ):
            build_dandelin_fixed_view(
                _construction("teaching-two-dimensional"),
                view="meridian",
                mode="depth_aware_teaching_transparent",
            )
        hyperbola = _construction(
            "teaching-hyperbola",
            axis_dot=0.2,
            model=ConeModel.OPEN_DOUBLE,
            axial_range=(-20.0, 20.0),
        )
        with self.assertRaisesRegex(
            DandelinFixedViewError,
            "cannot be certified|remains mixed|positive-area overlap",
        ):
            build_dandelin_fixed_view(
                hyperbola,
                view="spatial",
                projection_matrix=SPATIAL_MATRIX,
                mode="depth_aware_teaching_transparent",
            )

    def test_depth_aware_spatial_view_uses_certified_hidden_line_fragments(
        self,
    ) -> None:
        construction = _construction("automatic-hidden-lines")
        group = build_dandelin_fixed_view(
            construction,
            view="spatial",
            projection_matrix=SPATIAL_MATRIX,
            mode="depth_aware_diagrammatic",
        )

        self.assertEqual(group.mode, "depth_aware_diagrammatic")
        self.assertTrue(group.curve_visibility_authoritative)
        self.assertFalse(group.surface_visibility_authoritative)
        self.assertFalse(group.visibility_authoritative)
        self.assertGreater(group.metadata["hiddenSpanCount"], 0)
        self.assertEqual(
            group.metadata["tangentContactCount"],
            len(construction.spheres),
        )
        self.assertIsNotNone(group.visibility_frame)
        fragments = tuple(
            member
            for item in group.submobjects
            for member in item.submobjects
            if getattr(member, "dandelin_metadata", {}).get("renderIntent")
            in {"solid", "dashed"}
        )
        self.assertTrue(fragments)
        self.assertTrue(
            any(
                item.dandelin_metadata["renderIntent"] == "solid"
                for item in fragments
            )
        )
        self.assertTrue(
            any(
                item.dandelin_metadata["renderIntent"] == "dashed"
                and item.dandelin_metadata["occluderSurfaceIds"]
                for item in fragments
            )
        )
        hidden_z = [
            item.z_index
            for item in fragments
            if item.dandelin_metadata["renderIntent"] == "dashed"
        ]
        visible_z = [
            item.z_index
            for item in fragments
            if item.dandelin_metadata["renderIntent"] == "solid"
        ]
        surface_z = [
            member.z_index
            for item in group.submobjects
            for member in item.submobjects
            if getattr(member, "dandelin_metadata", {}).get("renderIntent")
            is None
            and getattr(member, "dandelin_metadata", {}).get("semanticKind")
            in {"cone_face", "section_plane", "dandelin_sphere"}
        ]
        self.assertLess(max(hidden_z), min(surface_z))
        self.assertLess(max(surface_z), min(visible_z))

    def test_depth_aware_mode_is_spatial_only_and_camera_dependent(self) -> None:
        construction = _construction("automatic-camera")
        with self.assertRaisesRegex(
            DandelinFixedViewError,
            "only valid for the spatial view",
        ):
            build_dandelin_fixed_view(
                construction,
                view="meridian",
                mode="depth_aware_diagrammatic",
            )
        rotated = np.asarray(
            (
                (0.8, -0.6, 0.0),
                (0.3, 0.4, -0.8660254037844386),
                (0.5196152422706632, 0.6928203230275509, 0.5),
            ),
            dtype=float,
        )
        first = build_dandelin_fixed_view(
            construction,
            view="spatial",
            projection_matrix=SPATIAL_MATRIX,
            mode="depth_aware_diagrammatic",
        )
        second = build_dandelin_fixed_view(
            construction,
            view="spatial",
            projection_matrix=rotated,
            mode="depth_aware_diagrammatic",
        )
        self.assertEqual(
            tuple(item.source_id for item in first.visibility_frame.strokes),
            tuple(item.source_id for item in second.visibility_frame.strokes),
        )
        self.assertNotEqual(
            first.visibility_frame.canonical_json(),
            second.visibility_frame.canonical_json(),
        )

    def test_three_views_are_finite_plain_vgroups_with_explicit_non_authority(
        self,
    ) -> None:
        construction = _construction()
        groups = {
            "spatial": build_dandelin_fixed_view(
                construction,
                view="spatial",
                projection_matrix=SPATIAL_MATRIX,
            ),
            "meridian": build_dandelin_fixed_view(
                construction,
                view="meridian",
            ),
            "section-plane": build_dandelin_fixed_view(
                construction,
                view="section-plane",
            ),
        }
        expected_flags = {
            "spatial": (True, True, True),
            "meridian": (True, False, True),
            "section-plane": (False, True, True),
        }

        for view, group in groups.items():
            with self.subTest(view=view):
                self.assertIs(type(group), VGroup)
                self.assertEqual(group.view, view)
                self.assertEqual(group.mode, "diagrammatic")
                self.assertIs(group.visibility_authoritative, False)
                self.assertEqual(group.metadata["view"], view)
                self.assertEqual(group.metadata["preset"], "classroom")
                self.assertIs(group.metadata["visibilityAuthoritative"], False)
                self.assertIs(group.metadata["sourceCoordinateUnits"], True)
                self.assertGreater(group.metadata["objectCount"], 0)
                self.assertEqual(
                    (
                        group.metadata["showContactCircles"],
                        group.metadata["showDirectrices"],
                        group.metadata["showFoci"],
                    ),
                    expected_flags[view],
                )
                bounds = np.asarray(group.metadata["finiteBounds"], dtype=float)
                self.assertEqual(bounds.shape, (4,))
                self.assertTrue(np.all(np.isfinite(bounds)))
                self.assertLess(bounds[0], bounds[2])
                self.assertLess(bounds[1], bounds[3])
                points = group.get_all_points()
                self.assertTrue(len(points))
                self.assertTrue(np.all(np.isfinite(points)))
                for item in _semantic_objects(group):
                    metadata = item.dandelin_metadata
                    self.assertTrue(metadata["semanticId"])
                    self.assertTrue(metadata["semanticSourceRefs"])

        spatial_kinds = {
            item.dandelin_metadata["semanticKind"]
            for item in _semantic_objects(groups["spatial"])
        }
        self.assertTrue(
            {
                "cone_surface",
                "section_plane",
                "section_curve",
                "sphere_surface",
                "contact_circle",
                "directrix",
                "focus",
            }.issubset(spatial_kinds)
        )
        meridian_kinds = {
            item.dandelin_metadata["semanticKind"]
            for item in _semantic_objects(groups["meridian"])
        }
        self.assertTrue(
            {
                "cone_face",
                "cone_generator",
                "section_line",
                "sphere_circle_section",
                "contact_circle_section_point",
                "focus",
            }.issubset(meridian_kinds)
        )
        section_kinds = {
            item.dandelin_metadata["semanticKind"]
            for item in _semantic_objects(groups["section-plane"])
        }
        self.assertTrue({"section_curve", "directrix", "focus"}.issubset(section_kinds))

    def test_spatial_spheres_use_the_general_affine_projection_ellipse(self) -> None:
        construction = _construction("affine-spheres")
        matrix = np.asarray(
            (
                (2.0, 0.5, 0.25),
                (0.2, 1.5, 0.75),
                (-0.1, 0.3, 1.0),
            ),
            dtype=float,
        )
        group = build_dandelin_fixed_view(
            construction,
            view="spatial",
            projection_matrix=matrix,
            show_contact_circles=False,
            show_directrices=False,
            show_foci=False,
        )

        spheres = _semantic_objects(group, "sphere_surface")
        self.assertEqual(len(spheres), len(construction.spheres))
        screen_singular = np.linalg.svd(matrix[:2], compute_uv=False)
        by_id = {item.sphere_id: item for item in construction.spheres}
        for mobject in spheres:
            metadata = mobject.submobjects[0].dandelin_metadata
            record = by_id[mobject.dandelin_metadata["semanticSourceRefs"][0]]
            expected_axes = record.sphere.radius * screen_singular
            np.testing.assert_allclose(
                metadata["semiAxes"],
                expected_axes,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                metadata["screenCenter"],
                matrix[:2] @ np.asarray(record.sphere.center),
                rtol=0.0,
                atol=1.0e-12,
            )
            basis = np.asarray(metadata["screenBasis"], dtype=float)
            np.testing.assert_allclose(
                basis @ basis.T,
                record.sphere.radius**2 * matrix[:2] @ matrix[:2].T,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            self.assertGreater(abs(float(basis[0, 1])), 0.0)

    def test_edge_on_contact_circles_become_finite_segments(self) -> None:
        construction = _construction("edge-on-contact")
        matrix = np.asarray(
            (
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0),
            ),
            dtype=float,
        )

        group = build_dandelin_fixed_view(
            construction,
            view="spatial",
            projection_matrix=matrix,
        )

        contacts = _semantic_objects(group, "contact_circle")
        self.assertEqual(len(contacts), len(construction.spheres))
        self.assertTrue(
            all(
                len(item.submobjects) == 1
                and isinstance(item.submobjects[0], Line)
                for item in contacts
            )
        )
        self.assertTrue(
            all(
                item.submobjects[0].dandelin_metadata["projectionRank"] == 1
                for item in contacts
            )
        )
        self.assertTrue(np.all(np.isfinite(group.get_all_points())))

    def test_circle_focus_sources_remain_distinct_at_one_visual_point_in_every_view(
        self,
    ) -> None:
        construction = _construction(
            "circle-focus",
            axis_dot=1.0,
            axial_range=(0.0, 10.0),
        )
        expected_refs = tuple(sorted(item.focus_id for item in construction.spheres))
        groups = (
            build_dandelin_fixed_view(
                construction,
                view="spatial",
                projection_matrix=SPATIAL_MATRIX,
            ),
            build_dandelin_fixed_view(construction, view="meridian"),
            build_dandelin_fixed_view(construction, view="section-plane"),
        )

        for group in groups:
            with self.subTest(view=group.view):
                foci = _semantic_objects(group, "focus")
                self.assertEqual(len(foci), 2)
                self.assertEqual(
                    tuple(
                        sorted(
                            item.dandelin_metadata["semanticSourceRefs"][0]
                            for item in foci
                        )
                    ),
                    expected_refs,
                )
                np.testing.assert_allclose(
                    foci[0].get_center(),
                    foci[1].get_center(),
                    rtol=0.0,
                    atol=1.0e-12,
                )

    def test_section_plane_never_draws_fake_sphere_or_contact_circles(self) -> None:
        for construction in (
            _construction("ellipse-section"),
            _construction(
                "circle-section",
                axis_dot=1.0,
                axial_range=(0.0, 10.0),
            ),
        ):
            with self.subTest(kind=construction.supporting_kind.value):
                with self.assertRaisesRegex(
                    DandelinFixedViewError,
                    "section-plane.*cannot show.*contact circles",
                ):
                    build_dandelin_fixed_view(
                        construction,
                        view="section-plane",
                        show_contact_circles=True,
                    )
                group = build_dandelin_fixed_view(
                    construction,
                    view="section-plane",
                )
                kinds = {
                    item.dandelin_metadata["semanticKind"]
                    for item in _semantic_objects(group)
                }
                self.assertNotIn("sphere_surface", kinds)
                self.assertNotIn("sphere_circle_section", kinds)
                self.assertNotIn("contact_circle", kinds)
                self.assertNotIn("contact_circle_section_point", kinds)
                self.assertIs(group.metadata["sectionPlaneSphereCircles"], False)

    def test_optional_flags_remove_only_their_supported_semantics(self) -> None:
        construction = _construction("flags")
        spatial = build_dandelin_fixed_view(
            construction,
            view="spatial",
            projection_matrix=SPATIAL_MATRIX,
            show_contact_circles=False,
            show_directrices=False,
            show_foci=False,
        )
        section = build_dandelin_fixed_view(
            construction,
            view="section-plane",
            show_contact_circles=False,
            show_directrices=False,
            show_foci=False,
        )
        meridian = build_dandelin_fixed_view(
            construction,
            view="meridian",
            show_contact_circles=False,
            show_directrices=False,
            show_foci=False,
        )

        for group in (spatial, section, meridian):
            kinds = {
                item.dandelin_metadata["semanticKind"]
                for item in _semantic_objects(group)
            }
            self.assertNotIn("focus", kinds)
            self.assertNotIn("directrix", kinds)
            self.assertNotIn("contact_circle", kinds)
            self.assertNotIn("contact_circle_section_point", kinds)
            self.assertTrue(all(group.metadata[name] is False for name in (
                "showContactCircles",
                "showDirectrices",
                "showFoci",
            )))

    def test_all_conic_families_have_deterministic_semantics_and_finite_bounds(
        self,
    ) -> None:
        cases = (
            (
                "ellipse",
                _construction("family-ellipse", axis_dot=0.8),
                DandelinConicFamily.ELLIPSE,
            ),
            (
                "parabola",
                _construction("family-parabola", axis_dot=0.5),
                DandelinConicFamily.PARABOLA,
            ),
            (
                "hyperbola",
                _construction(
                    "family-hyperbola",
                    axis_dot=0.2,
                    model=ConeModel.OPEN_DOUBLE,
                    axial_range=(-20.0, 20.0),
                ),
                DandelinConicFamily.HYPERBOLA,
            ),
            (
                "circle",
                _construction(
                    "family-circle",
                    axis_dot=1.0,
                    axial_range=(0.0, 10.0),
                ),
                DandelinConicFamily.ELLIPSE,
            ),
        )
        for label, construction, family in cases:
            self.assertIs(construction.family, family)
            for view in ("spatial", "meridian", "section-plane"):
                with self.subTest(kind=label, view=view):
                    kwargs = (
                        {"projection_matrix": SPATIAL_MATRIX}
                        if view == "spatial"
                        else {}
                    )
                    first = build_dandelin_fixed_view(
                        construction,
                        view=view,
                        **kwargs,
                    )
                    second = build_dandelin_fixed_view(
                        construction,
                        view=view,
                        **kwargs,
                    )
                    self.assertEqual(_semantic_signature(first), _semantic_signature(second))
                    self.assertEqual(
                        first.metadata["finiteBounds"],
                        second.metadata["finiteBounds"],
                    )
                    self.assertEqual(first.metadata["family"], family.value)
                    self.assertTrue(np.all(np.isfinite(first.get_all_points())))

    def test_depth_aware_spatial_view_supports_every_dandelin_family(self) -> None:
        cases = (
            _construction("automatic-family-ellipse", axis_dot=0.8),
            _construction("automatic-family-parabola", axis_dot=0.5),
            _construction(
                "automatic-family-hyperbola",
                axis_dot=0.2,
                model=ConeModel.OPEN_DOUBLE,
                axial_range=(-20.0, 20.0),
            ),
            _construction(
                "automatic-family-circle",
                axis_dot=1.0,
                axial_range=(0.0, 10.0),
            ),
        )
        for construction in cases:
            with self.subTest(kind=construction.supporting_kind.value):
                group = build_dandelin_fixed_view(
                    construction,
                    view="spatial",
                    projection_matrix=SPATIAL_MATRIX,
                    mode="depth_aware_diagrammatic",
                )
                self.assertTrue(group.curve_visibility_authoritative)
                self.assertFalse(group.surface_visibility_authoritative)
                self.assertGreater(group.metadata["hiddenSpanCount"], 0)
                self.assertEqual(
                    group.metadata["tangentContactCount"],
                    len(construction.spheres),
                )
                self.assertTrue(np.all(np.isfinite(group.get_all_points())))

    def test_invalid_views_matrices_presets_and_flags_fail_closed(self) -> None:
        construction = _construction("invalid-inputs")
        with self.assertRaisesRegex(TypeError, "DandelinConstruction3D"):
            build_dandelin_fixed_view(object(), view="meridian")
        for view in ("", "front", None):
            with self.subTest(view=view):
                with self.assertRaises((DandelinFixedViewError, TypeError)):
                    build_dandelin_fixed_view(construction, view=view)
        with self.assertRaisesRegex(DandelinFixedViewError, "preset"):
            build_dandelin_fixed_view(
                construction,
                view="meridian",
                preset="physical",
            )
        with self.assertRaisesRegex(DandelinFixedViewError, "requires"):
            build_dandelin_fixed_view(construction, view="spatial")
        with self.assertRaisesRegex(DandelinFixedViewError, "only valid"):
            build_dandelin_fixed_view(
                construction,
                view="meridian",
                projection_matrix=np.eye(3),
            )
        bad_matrices = (
            np.eye(2),
            ((True, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            ((1.0, 0.0, 0.0), (0.0, float("nan"), 0.0), (0.0, 0.0, 1.0)),
        )
        for matrix in bad_matrices:
            with self.subTest(matrix=repr(matrix)):
                with self.assertRaises(DandelinFixedViewError):
                    build_dandelin_fixed_view(
                        construction,
                        view="spatial",
                        projection_matrix=matrix,
                    )
        for name in (
            "show_contact_circles",
            "show_directrices",
            "show_foci",
        ):
            for invalid in (1, np.bool_(True), "yes"):
                with self.subTest(name=name, invalid=repr(invalid)):
                    with self.assertRaisesRegex(TypeError, "must be a bool or None"):
                        build_dandelin_fixed_view(
                            construction,
                            view="meridian",
                            **{name: invalid},
                        )

        with self.assertRaisesRegex(
            DandelinFixedViewError,
            "meridian.*cannot show.*directrix",
        ):
            build_dandelin_fixed_view(
                construction,
                view="meridian",
                show_directrices=True,
            )

    def test_module_has_no_compiler_scene_or_quadric_manim_facade_dependency(self) -> None:
        source = inspect.getsource(fixed_view_module)
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertFalse(any("compiler" in name for name in imported))
        self.assertFalse(any("dandelin_authoring" in name for name in imported))
        self.assertFalse(any(name.endswith("quadrics.manim") for name in imported))
        self.assertNotIn("DandelinSection3D", source)
        referenced_names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        self.assertNotIn("Scene", referenced_names)
        self.assertEqual(
            Path(fixed_view_module.__file__).name,
            "dandelin_fixed_view.py",
        )


if __name__ == "__main__":
    unittest.main()
