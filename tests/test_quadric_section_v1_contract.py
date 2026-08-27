"""Executable release contract for the finite-cone section v1 baseline."""

from __future__ import annotations

import importlib
import json
from math import pi
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from manim import RendererType, Scene

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.compositing import compute_quadric_compositing
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    PlaneDisplayPatchSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.manim import (
    QuadricManimError,
    QuadricManimLimits,
    QuadricOcclusion3D,
)
from polyhedron_visibility.quadrics.projection import build_opaque_projection_proxy
from polyhedron_visibility.quadrics.section_compositing import (
    compute_quadric_section_compositing,
)
from polyhedron_visibility.quadrics.visibility import compute_quadric_visibility
from tikz_native.version import (
    COMPONENT_QUADRIC_GEOMETRY,
    COMPONENT_QUADRIC_MANIM,
    COMPONENT_QUADRIC_VISIBILITY,
    provider_component_contract_revisions,
    provider_component_revisions,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "tests" / "fixtures" / "quadric-section-v1-contract.json"
CAIRO_BASELINE_PATH = (
    ROOT / "tests" / "baselines" / "quadric-section-v1-cairo.json"
)


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as source:
        result = json.load(source)
    if not isinstance(result, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return result


def _small_limits() -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=2,
        max_curves=1,
        max_fragments_per_curve=2,
        max_segments_per_fragment=8,
        max_surface_segments=16,
        max_dashes_per_fragment=2,
        max_projected_length=8.0,
        max_total_mobjects=256,
        max_boundary_sources=8,
        max_boundary_styles=8,
    )


class QuadricSectionV1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = _load_json(CONTRACT_PATH)
        cls.cairo_baseline = _load_json(CAIRO_BASELINE_PATH)

    def test_support_matrix_is_complete_and_frozen(self) -> None:
        self.assertEqual(
            self.contract["schema"],
            "manim-quadric-section-support-contract/v1",
        )
        self.assertEqual(self.contract["contract_id"], "quadric-section-v1")
        self.assertEqual(
            self.contract["baseline_main_commit"],
            "34f50e9b2ab66a969ae6a0648669be238ff943ae",
        )
        expected = {
            "closed_finite_single_cone": "supported",
            "open_finite_single_cone_shell": "supported",
            "finite_cone_frustum_section_and_occlusion": "supported",
            "frustum_two_cap_component_shading": (
                "unsupported_explicit_failure"
            ),
            "open_double_display_and_general_occlusion": (
                "supported_with_constraints"
            ),
            "open_double_unified_section_plane_compositing": (
                "unsupported_explicit_failure"
            ),
            "one_surface_one_cutting_plane": "supported",
            "multiple_intersecting_surfaces_one_cutting_plane": (
                "unsupported_explicit_failure"
            ),
            "parallel_projection": "supported",
            "perspective_projection": "unsupported",
            "manim_cairo": "supported",
            "manim_opengl": "unsupported_explicit_failure",
            "cutting_plane_with_two_dimensional_projection": "supported",
            "edge_on_cutting_plane": "unsupported_explicit_failure",
        }
        actual = {
            item["id"]: item["status"]
            for item in self.contract["support_matrix"]
        }
        self.assertEqual(actual, expected)

    def test_component_contract_and_render_revisions_match_provider(self) -> None:
        component_names = {
            "quadric_geometry": COMPONENT_QUADRIC_GEOMETRY,
            "quadric_visibility": COMPONENT_QUADRIC_VISIBILITY,
            "quadric_manim": COMPONENT_QUADRIC_MANIM,
        }
        contract_revisions = provider_component_contract_revisions()
        render_revisions = provider_component_revisions()
        frozen = self.contract["components"]
        for fixture_name, component_name in component_names.items():
            with self.subTest(component=fixture_name):
                self.assertEqual(
                    frozen[fixture_name]["contract_revision"],
                    contract_revisions[component_name],
                )
                self.assertEqual(
                    frozen[fixture_name]["render_revision"],
                    render_revisions[component_name],
                )

    def test_release_evidence_names_resolve_to_real_tests(self) -> None:
        evidence = {
            name
            for fixture in self.contract["release_fixtures"]
            for name in fixture["evidence_tests"]
        }
        for dotted_name in sorted(evidence):
            with self.subTest(test=dotted_name):
                module_name, class_name, method_name = dotted_name.rsplit(".", 2)
                module = importlib.import_module(module_name)
                case = getattr(module, class_name)
                self.assertTrue(issubclass(case, unittest.TestCase))
                self.assertTrue(callable(getattr(case, method_name)))

        self.assertEqual(
            self.cairo_baseline["contract_id"],
            self.contract["contract_id"],
        )
        self.assertEqual(self.cairo_baseline["renderer"], "cairo")
        for fixture in self.cairo_baseline["fixtures"].values():
            self.assertIn(fixture["test"], evidence)
            profile = self.cairo_baseline["profiles"][fixture["profile"]]
            self.assertGreater(profile["pixel_width"], 0)
            self.assertGreater(profile["pixel_height"], 0)
            self.assertGreater(profile["frame_rate"], 0)

    def test_exact_side_fixture_executes_rank_one_rim_path(self) -> None:
        fixture = next(
            item
            for item in self.contract["release_fixtures"]
            if item["id"] == "exact_side_open_cone_and_frustum_trim_rims"
        )
        geometry = fixture["geometry"]
        surface_data = geometry["surface"]
        plane_data = geometry["plane"]
        view = ParallelView.from_matrix(
            self.contract["projections"][geometry["projection"]]
        )
        plane = SectionPlane(
            "v1-side-view-cut",
            plane_data["point"],
            plane_data["normal"],
            u_axis=plane_data["u_axis"],
        )
        patch_spec = PlaneDisplayPatchSpec(
            "v1-side-view-patch",
            plane.plane_id,
            3.0,
            3.0,
        )
        expected = fixture["expected"]
        for index, axial_range in enumerate(surface_data["axial_ranges"]):
            with self.subTest(axial_range=axial_range):
                cone = ConeSpec(
                    f"v1-side-cone-{index}",
                    surface_data["apex"],
                    surface_data["axis"],
                    surface_data.get("half_angle_radians", pi / 4.0),
                    axial_range,
                    radial_axis=surface_data["radial_axis"],
                    model=ConeModel(surface_data["model"]),
                )
                proxy = build_opaque_projection_proxy(
                    cone,
                    view,
                    max_chord_error=0.008,
                    max_segments=768,
                )
                base = compute_quadric_compositing(
                    compute_quadric_visibility((), (cone,), view),
                    (proxy,),
                )
                frame = compute_quadric_section_compositing(
                    base,
                    cone,
                    plane,
                    patch_spec,
                    view,
                    max_screen_error=0.08,
                )
                self.assertEqual(
                    len(proxy.vertices),
                    expected["proxy_vertex_counts"][index],
                )
                self.assertEqual(
                    len(cone.trim_rims),
                    expected["trim_rim_counts"][index],
                )
                self.assertTrue(frame.plane_fragments)

    def test_unsupported_binding_boundaries_fail_before_scene_mutation(self) -> None:
        plane = SectionPlane(
            "v1-cut",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        double = ConeSpec(
            "v1-double",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 4.0,
            (-2.0, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.OPEN_DOUBLE,
        )
        with self.assertRaisesRegex(
            QuadricManimError,
            "section compositing requires exactly one",
        ):
            QuadricOcclusion3D(
                Scene(),
                surfaces=(double,),
                curves=(),
                section_plane=plane,
                limits=_small_limits(),
            )

        with self.assertRaisesRegex(
            QuadricManimError,
            "section compositing requires exactly one",
        ):
            QuadricOcclusion3D(
                Scene(),
                surfaces=(
                    SphereSpec("v1-sphere-a", (-2.0, 0.0, 0.0), 1.0),
                    SphereSpec("v1-sphere-b", (2.0, 0.0, 0.0), 1.0),
                ),
                curves=(),
                section_plane=plane,
                limits=_small_limits(),
            )

        scene = Scene()
        controller = QuadricOcclusion3D(
            scene,
            surfaces=(SphereSpec("v1-opengl-sphere", (0.0, 0.0, 0.0), 1.0),),
            curves=(),
            limits=_small_limits(),
        )
        fake_config = SimpleNamespace(renderer=RendererType.OPENGL)
        with patch("polyhedron_visibility.quadrics.manim.config", fake_config):
            with self.assertRaisesRegex(
                QuadricManimError,
                "supports the Cairo renderer only",
            ):
                controller.attach()
        self.assertEqual(scene.mobjects, [])

    def test_public_docs_link_the_frozen_contract(self) -> None:
        for relative in (
            "README.md",
            "README.zh-CN.md",
            "docs/quadric-occlusion.md",
        ):
            with self.subTest(document=relative):
                text = (ROOT / relative).read_text(encoding="utf-8")
                self.assertIn("quadric-section-v1-contract.md", text)


if __name__ == "__main__":
    unittest.main()
