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
