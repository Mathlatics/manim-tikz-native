from __future__ import annotations

import ast
import copy
import inspect
import json
from math import pi
import unittest

import numpy as np

from polyhedron_visibility.quadrics.contract import ConeModel
from polyhedron_visibility.quadrics.planar_curves import PlanarFrame3D
import tikz_native.dandelin_contract as contract_module
from tikz_native.dandelin_contract import (
    TIKZ_DANDELIN_CONSTRUCTION_3D_SCHEMA,
    TIKZ_DANDELIN_SPATIAL_VIEW_SCHEMA,
    TIKZ_DANDELIN_STATIC_DIAGRAM_V1_SCHEMA,
    TIKZ_DANDELIN_STATIC_DIAGRAM_SCHEMA,
    TIKZ_SPACE_RIGHT_CONE_3D_SCHEMA,
    TikzDandelinContractError,
    build_dandelin_construction_contract,
    build_dandelin_static_diagram_contract,
    build_space_right_cone_contract,
    canonical_dandelin_contract_json,
    restore_dandelin_construction_contract,
    restore_dandelin_static_diagram_contract,
    restore_space_right_cone_contract,
    section_plane_from_planar_frame,
)


class TikzDandelinContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.coordinates = {
            "A": (-0.0, 0.0, -0.0),
            "Z": (0.0, 0.0, 2.0),
            # The axial component must be discarded.  Only +x authors the
            # canonical radial direction.
            "R": (2.0, 0.0, 1.0),
        }
        cls.cone_contract = build_space_right_cone_contract(
            "cone",
            ("A", "Z", "R"),
            cls.coordinates,
            30,
            (0, 20),
            ConeModel.OPEN_SINGLE,
        )
        cls.plane_frame = PlanarFrame3D(
            "cut",
            (0.0, 0.0, 2.0),
            (0.6, 0.0, 0.8),
            (0.0, 1.0, 0.0),
        )
        cls.construction_contract = build_dandelin_construction_contract(
            "dan",
            cone_ref="cone",
            cone=cls.cone_contract.cone,
            plane_ref="cut",
            plane_frame=cls.plane_frame,
        )

    def test_named_points_certify_finite_right_cone_and_round_trip(self) -> None:
        record = self.cone_contract
        cone = record.cone

        self.assertEqual(cone.apex, (0.0, 0.0, 0.0))
        np.testing.assert_allclose(cone.axis, (0.0, 0.0, 1.0), atol=0.0)
        np.testing.assert_allclose(cone.radial_axis, (1.0, 0.0, 0.0), atol=0.0)
        self.assertEqual(cone.half_angle, pi / 6.0)
        self.assertEqual(cone.axial_range, (0.0, 20.0))
        self.assertIs(cone.model, ConeModel.OPEN_SINGLE)

        payload = record.to_dict()
        self.assertEqual(payload["schema"], TIKZ_SPACE_RIGHT_CONE_3D_SCHEMA)
        self.assertEqual(payload["pointNames"], ["A", "Z", "R"])
        self.assertIs(payload["static"], True)
        self.assertEqual(
            restore_space_right_cone_contract(
                payload,
                self.coordinates,
                expected_cone_ref="cone",
            ),
            record,
        )
        encoded = record.canonical_json()
        self.assertEqual(json.loads(encoded), payload)
        self.assertNotIn("-0.0", encoded)

    def test_oblique_radial_seed_is_projected_before_normalization(self) -> None:
        apex = np.asarray((3.0, -2.0, 1.0))
        axis_seed = np.asarray((1.0, 2.0, 3.0))
        radial_seed = np.asarray((2.0, -1.0, 0.0))
        coordinates = {
            "A": apex,
            "Z": apex + axis_seed,
            "R": apex + 5.0 * axis_seed + radial_seed,
        }

        record = build_space_right_cone_contract(
            "oblique-cone",
            ("A", "Z", "R"),
            coordinates,
            25.0,
            (-8.0, 8.0),
            "open_double",
        )

        axis = np.asarray(record.cone.axis)
        radial = np.asarray(record.cone.radial_axis)
        self.assertAlmostEqual(float(np.dot(axis, radial)), 0.0, places=14)
        self.assertAlmostEqual(float(np.linalg.norm(axis)), 1.0, places=14)
        self.assertAlmostEqual(float(np.linalg.norm(radial)), 1.0, places=14)
        self.assertGreater(float(np.dot(radial, radial_seed)), 0.0)

    def test_cone_author_data_rejects_boolean_nonfinite_and_degenerate_values(
        self,
    ) -> None:
        base = dict(
            cone_ref="bad-cone",
            point_names=("A", "Z", "R"),
            coordinates={
                "A": (0.0, 0.0, 0.0),
                "Z": (0.0, 0.0, 1.0),
                "R": (1.0, 0.0, 0.0),
            },
            half_angle_degrees=30.0,
            axial_range=(0.0, 9.0),
            model="open_single",
        )
        cases = (
            {"half_angle_degrees": True},
            {"half_angle_degrees": float("nan")},
            {"half_angle_degrees": 0.0},
            {"half_angle_degrees": 90.0},
            {"axial_range": (0.0, True)},
            {"axial_range": (0.0, float("inf"))},
            {"axial_range": (2.0, 1.0)},
            {"model": True},
            {"model": "analytic_double", "axial_range": (-2.0, 2.0)},
            {"point_names": ("A", "A", "R")},
            {
                "coordinates": {
                    "A": (0.0, 0.0, 0.0),
                    "Z": (0.0, 0.0, 0.0),
                    "R": (1.0, 0.0, 0.0),
                }
            },
            {
                "coordinates": {
                    "A": (0.0, 0.0, 0.0),
                    "Z": (0.0, 0.0, 1.0),
                    "R": (0.0, 0.0, 3.0),
                }
            },
            {
                "coordinates": {
                    "A": (False, 0.0, 0.0),
                    "Z": (0.0, 0.0, 1.0),
                    "R": (1.0, 0.0, 0.0),
                }
            },
            {
                "coordinates": {
                    "A": (0.0, 0.0, 0.0),
                    "Z": (0.0, 0.0, float("inf")),
                    "R": (1.0, 0.0, 0.0),
                }
            },
        )
        for changes in cases:
            with self.subTest(changes=changes):
                arguments = {**base, **changes}
                with self.assertRaises(TikzDandelinContractError):
                    build_space_right_cone_contract(**arguments)

    def test_cone_restore_recomputes_and_rejects_tampering(self) -> None:
        payload = self.cone_contract.to_dict()
        cases: list[dict[str, object]] = []

        forged_axis = copy.deepcopy(payload)
        forged_axis["cone"]["axis"][0] = 0.25
        cases.append(forged_axis)

        extra = copy.deepcopy(payload)
        extra["guessed"] = True
        cases.append(extra)

        not_static = copy.deepcopy(payload)
        not_static["static"] = 1
        cases.append(not_static)

        tuple_names = copy.deepcopy(payload)
        tuple_names["pointNames"] = ("A", "Z", "R")
        cases.append(tuple_names)

        for forged in cases:
            with self.subTest(forged=forged):
                with self.assertRaises(TikzDandelinContractError):
                    restore_space_right_cone_contract(forged, self.coordinates)

        changed_coordinates = dict(self.coordinates)
        changed_coordinates["Z"] = (0.0, 1.0, 2.0)
        with self.assertRaisesRegex(
            TikzDandelinContractError,
            "stale|forged|canonical",
        ):
            restore_space_right_cone_contract(payload, changed_coordinates)

    def test_certified_frame_builds_section_plane_and_construction(self) -> None:
        plane = section_plane_from_planar_frame(
            self.plane_frame,
            expected_plane_ref="cut",
        )
        self.assertEqual(plane.point, self.plane_frame.point)
        self.assertEqual(plane.normal, self.plane_frame.normal)
        self.assertEqual(plane.u_axis, self.plane_frame.u_axis)

        record = self.construction_contract
        construction = record.construction
        self.assertEqual(construction.construction_id, "dan")
        self.assertEqual(construction.cone, self.cone_contract.cone)
        self.assertEqual(construction.plane, plane)
        self.assertEqual(construction.family.value, "ellipse")
        self.assertEqual(len(construction.spheres), 2)

        payload = record.to_dict()
        self.assertEqual(
            payload["schema"],
            TIKZ_DANDELIN_CONSTRUCTION_3D_SCHEMA,
        )
        self.assertEqual(payload["constructionRef"], "dan")
        self.assertEqual(payload["coneRef"], "cone")
        self.assertEqual(payload["planeRef"], "cut")
        self.assertEqual(
            restore_dandelin_construction_contract(
                payload,
                cone=self.cone_contract.cone,
                plane_frame=self.plane_frame,
                expected_construction_ref="dan",
                expected_cone_ref="cone",
                expected_plane_ref="cut",
            ),
            record,
        )

        oblique = PlanarFrame3D(
            "oblique-cut",
            (1.0, -2.0, 3.0),
            (1.0, 2.0, 3.0),
            (2.0, -1.0, 0.5),
        )
        oblique_plane = section_plane_from_planar_frame(oblique)
        np.testing.assert_allclose(
            oblique_plane.normal,
            oblique.normal,
            rtol=0.0,
            atol=64.0 * np.finfo(float).eps,
        )
        np.testing.assert_allclose(
            oblique_plane.u_axis,
            oblique.u_axis,
            rtol=0.0,
            atol=64.0 * np.finfo(float).eps,
        )

    def test_construction_restore_rejects_nested_and_authority_tampering(self) -> None:
        payload = self.construction_contract.to_dict()
        cases: list[dict[str, object]] = []

        sphere = copy.deepcopy(payload)
        sphere["construction"]["spheres"][0]["sphere"]["radius"] *= 1.01
        cases.append(sphere)

        wrong_ref = copy.deepcopy(payload)
        wrong_ref["coneRef"] = "another-cone"
        cases.append(wrong_ref)

        extra = copy.deepcopy(payload)
        extra["implementationMode"] = "physical"
        cases.append(extra)

        not_static = copy.deepcopy(payload)
        not_static["static"] = 1
        cases.append(not_static)

        for forged in cases:
            with self.subTest(forged=forged):
                with self.assertRaises(TikzDandelinContractError):
                    restore_dandelin_construction_contract(
                        forged,
                        cone=self.cone_contract.cone,
                        plane_frame=self.plane_frame,
                    )

        moved_frame = PlanarFrame3D(
            "cut",
            (0.0, 0.0, 2.25),
            self.plane_frame.normal,
            self.plane_frame.u_axis,
        )
        with self.assertRaises(TikzDandelinContractError):
            restore_dandelin_construction_contract(
                payload,
                cone=self.cone_contract.cone,
                plane_frame=moved_frame,
            )

    def test_all_three_restore_layers_use_strict_json_numeric_types_and_zero_signs(
        self,
    ) -> None:
        cone = copy.deepcopy(self.cone_contract.to_dict())
        cone["cone"]["axis"][2] = True
        with self.assertRaisesRegex(
            TikzDandelinContractError,
            "stale|forged|canonical",
        ):
            restore_space_right_cone_contract(cone, self.coordinates)

        signed_zero = copy.deepcopy(self.cone_contract.to_dict())
        signed_zero["cone"]["apex"][0] = -0.0
        with self.assertRaisesRegex(
            TikzDandelinContractError,
            "stale|forged|canonical",
        ):
            restore_space_right_cone_contract(signed_zero, self.coordinates)

        construction = copy.deepcopy(self.construction_contract.to_dict())
        construction["construction"]["cone"]["axis"][2] = True
        with self.assertRaisesRegex(
            TikzDandelinContractError,
            "stale|forged|canonical",
        ):
            restore_dandelin_construction_contract(
                construction,
                cone=self.cone_contract.cone,
                plane_frame=self.plane_frame,
            )

        diagram = build_dandelin_static_diagram_contract(
            self.construction_contract.construction,
            view="spatial",
        ).to_dict()
        diagram["viewGeometry"]["construction"]["cone"]["axis"][2] = True
        with self.assertRaisesRegex(
            TikzDandelinContractError,
            "stale|forged|canonical",
        ):
            restore_dandelin_static_diagram_contract(
                diagram,
                self.construction_contract.construction,
            )

    def test_three_static_views_have_canonical_policy_and_local_ids(self) -> None:
        construction = self.construction_contract.construction
        expected_geometry_schemas = {
            "spatial": TIKZ_DANDELIN_SPATIAL_VIEW_SCHEMA,
            "meridian": "manim-dandelin-meridian-diagram-2d/v1",
            "section-plane": "manim-dandelin-section-plane-diagram-2d/v1",
        }
        for view in ("spatial", "meridian", "section-plane"):
            with self.subTest(view=view):
                diagram = build_dandelin_static_diagram_contract(
                    construction,
                    view=view,
                )
                payload = diagram.to_dict()
                self.assertEqual(
                    payload["schema"],
                    TIKZ_DANDELIN_STATIC_DIAGRAM_SCHEMA,
                )
                self.assertEqual(payload["diagramId"], f"dan:view:{view}")
                self.assertEqual(payload["constructionRef"], "dan")
                self.assertEqual(payload["coneRef"], "cone")
                self.assertEqual(payload["planeRef"], "cut")
                self.assertEqual(payload["mode"], "diagrammatic")
                self.assertIs(payload["visibilityAuthoritative"], False)
                self.assertIs(payload["static"], True)
                self.assertEqual(payload["preset"], "classroom")
                self.assertEqual(
                    payload["viewGeometry"]["schema"],
                    expected_geometry_schemas[view],
                )
                object_ids = [item["id"] for item in payload["semanticObjects"]]
                self.assertEqual(len(object_ids), len(set(object_ids)))
                self.assertTrue(
                    all(item.startswith(f"dan:view:{view}:object:") for item in object_ids)
                )
                self.assertTrue(
                    all(
                        item["role"] and item["sourceRef"]
                        for item in payload["semanticObjects"]
                    )
                )
                self.assertEqual(
                    restore_dandelin_static_diagram_contract(
                        payload,
                        construction,
                        expected_diagram_id=f"dan:view:{view}",
                    ),
                    diagram,
                )

        section_geometry = build_dandelin_static_diagram_contract(
            construction,
            view="section-plane",
        ).view_geometry
        self.assertNotIn("sphereCircles", section_geometry)
        self.assertNotIn("circles", section_geometry)

    def test_source_refs_link_corresponding_geometry_across_views(self) -> None:
        construction = self.construction_contract.construction
        spatial = build_dandelin_static_diagram_contract(
            construction,
            view="spatial",
        )
        meridian = build_dandelin_static_diagram_contract(
            construction,
            view="meridian",
        )
        section = build_dandelin_static_diagram_contract(
            construction,
            view="section-plane",
        )

        def refs(diagram, role: str) -> set[str]:
            return {
                item.source_ref
                for item in diagram.semantic_objects
                if item.role == role
            }

        focus_ids = {item.focus_id for item in construction.spheres}
        sphere_ids = {item.sphere_id for item in construction.spheres}
        directrix_ids = {item.directrix_id for item in construction.directrices}
        self.assertEqual(refs(spatial, "focus"), focus_ids)
        self.assertEqual(refs(meridian, "focus"), focus_ids)
        self.assertEqual(refs(section, "focus"), focus_ids)
        self.assertEqual(refs(spatial, "sphere_surface"), sphere_ids)
        self.assertEqual(refs(meridian, "sphere_circle_section"), sphere_ids)
        self.assertEqual(refs(spatial, "directrix"), directrix_ids)
        self.assertEqual(refs(section, "directrix"), directrix_ids)
        self.assertTrue(
            set(item.object_id for item in spatial.semantic_objects).isdisjoint(
                item.object_id for item in meridian.semantic_objects
            )
        )

    def test_depth_aware_mode_round_trips_only_for_spatial_view(self) -> None:
        construction = self.construction_contract.construction
        diagram = build_dandelin_static_diagram_contract(
            construction,
            view="spatial",
            mode="depth_aware_diagrammatic",
        )
        payload = diagram.to_dict()
        self.assertEqual(payload["mode"], "depth_aware_diagrammatic")
        self.assertIs(payload["visibilityAuthoritative"], False)
        self.assertIs(payload["curveVisibilityAuthoritative"], True)
        self.assertIs(payload["surfaceVisibilityAuthoritative"], False)
        self.assertEqual(
            restore_dandelin_static_diagram_contract(payload, construction),
            diagram,
        )
        for view in ("meridian", "section-plane"):
            with self.subTest(view=view):
                with self.assertRaisesRegex(
                    TikzDandelinContractError,
                    "only valid for the spatial view",
                ):
                    build_dandelin_static_diagram_contract(
                        construction,
                        view=view,
                        mode="depth_aware_diagrammatic",
                    )

    def test_legacy_v1_diagram_payload_restores_to_current_contract(self) -> None:
        construction = self.construction_contract.construction
        current = build_dandelin_static_diagram_contract(
            construction,
            view="spatial",
        )
        legacy = current.to_dict()
        legacy["schema"] = TIKZ_DANDELIN_STATIC_DIAGRAM_V1_SCHEMA
        del legacy["curveVisibilityAuthoritative"]
        del legacy["surfaceVisibilityAuthoritative"]

        restored = restore_dandelin_static_diagram_contract(
            legacy,
            construction,
        )
        self.assertEqual(restored, current)
        self.assertEqual(
            restored.to_dict()["schema"],
            TIKZ_DANDELIN_STATIC_DIAGRAM_SCHEMA,
        )

        forged = copy.deepcopy(legacy)
        forged["mode"] = "depth_aware_diagrammatic"
        with self.assertRaisesRegex(
            TikzDandelinContractError,
            "v1 requires mode=diagrammatic",
        ):
            restore_dandelin_static_diagram_contract(forged, construction)

    def test_flags_filter_drawables_without_changing_certified_view_geometry(
        self,
    ) -> None:
        construction = self.construction_contract.construction
        full = build_dandelin_static_diagram_contract(
            construction,
            view="spatial",
        )
        bare = build_dandelin_static_diagram_contract(
            construction,
            view="spatial",
            show_contact_circles=False,
            show_foci=False,
            show_directrices=False,
        )
        self.assertEqual(full.view_geometry, bare.view_geometry)
        self.assertEqual(
            {item.role for item in bare.semantic_objects},
            {"cone_surface", "section_plane", "section_curve", "sphere_surface"},
        )

        meridian = build_dandelin_static_diagram_contract(
            construction,
            view="meridian",
            show_contact_circles=False,
            show_foci=False,
        )
        meridian_roles = {item.role for item in meridian.semantic_objects}
        self.assertIn("sphere_circle_section", meridian_roles)
        self.assertNotIn("contact_circle_section_point", meridian_roles)
        section = build_dandelin_static_diagram_contract(
            construction,
            view="section-plane",
            show_foci=False,
            show_directrices=False,
        )
        self.assertEqual(
            {item.role for item in section.semantic_objects},
            {"section_curve"},
        )

    def test_section_plane_rejects_fake_contact_circles_and_bad_options(self) -> None:
        construction = self.construction_contract.construction
        with self.assertRaisesRegex(
            TikzDandelinContractError,
            "section-plane.*cannot display|not geometry",
        ):
            build_dandelin_static_diagram_contract(
                construction,
                view="section-plane",
                show_contact_circles=True,
            )
        for kwargs in (
            {"view": "perspective"},
            {"view": "spatial", "preset": "physical"},
            {"view": "spatial", "show_foci": 1},
            {"view": "spatial", "show_directrices": np.bool_(True)},
            {"view": "meridian", "show_directrices": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(TikzDandelinContractError):
                    build_dandelin_static_diagram_contract(
                        construction,
                        **kwargs,
                    )

    def test_diagram_restore_rejects_policy_geometry_and_semantic_tampering(
        self,
    ) -> None:
        construction = self.construction_contract.construction
        payload = build_dandelin_static_diagram_contract(
            construction,
            view="meridian",
        ).to_dict()
        cases: list[dict[str, object]] = []

        physical = copy.deepcopy(payload)
        physical["mode"] = "physical"
        cases.append(physical)

        authoritative = copy.deepcopy(payload)
        authoritative["visibilityAuthoritative"] = True
        cases.append(authoritative)

        curve_authority = copy.deepcopy(payload)
        curve_authority["curveVisibilityAuthoritative"] = True
        cases.append(curve_authority)

        surface_authority = copy.deepcopy(payload)
        surface_authority["surfaceVisibilityAuthoritative"] = True
        cases.append(surface_authority)

        not_static = copy.deepcopy(payload)
        not_static["static"] = False
        cases.append(not_static)

        semantic = copy.deepcopy(payload)
        semantic["semanticObjects"][0]["sourceRef"] = "forged-source"
        cases.append(semantic)

        geometry = copy.deepcopy(payload)
        geometry["viewGeometry"]["sectionLine"]["worldPoint"][0] += 0.125
        cases.append(geometry)

        bad_flag = copy.deepcopy(payload)
        bad_flag["flags"]["showFoci"] = 1
        cases.append(bad_flag)

        extra = copy.deepcopy(payload)
        extra["cameraShots"] = []
        cases.append(extra)

        for forged in cases:
            with self.subTest(forged=forged):
                with self.assertRaises(TikzDandelinContractError):
                    restore_dandelin_static_diagram_contract(
                        forged,
                        construction,
                    )

        other = build_dandelin_construction_contract(
            "other-dan",
            cone_ref="cone",
            cone=self.cone_contract.cone,
            plane_ref="cut",
            plane_frame=self.plane_frame,
        ).construction
        with self.assertRaises(TikzDandelinContractError):
            restore_dandelin_static_diagram_contract(payload, other)

    def test_canonical_json_removes_signed_zero_and_rejects_nan(self) -> None:
        encoded = canonical_dandelin_contract_json(
            {"point": [-0.0, 1.0], "nested": {"value": -0.0}}
        )
        self.assertEqual(encoded, '{"nested":{"value":0.0},"point":[0.0,1.0]}')
        self.assertNotIn("-0.0", encoded)
        with self.assertRaises(TikzDandelinContractError):
            canonical_dandelin_contract_json({"value": float("nan")})

    def test_module_is_renderer_neutral(self) -> None:
        tree = ast.parse(inspect.getsource(contract_module))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertFalse(
            any(name == "manim" or name.startswith("manim.") for name in imported)
        )
        self.assertNotIn("tikz_native.compiler", imported)
        self.assertNotIn("compiler", imported)


if __name__ == "__main__":
    unittest.main()
