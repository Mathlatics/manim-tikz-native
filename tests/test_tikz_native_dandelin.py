from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tikz_native import compile_document
from tikz_native.animation import semantic_animation_layers
from tikz_native.provider import (
    TikzNativeProviderError,
    compile_asset,
    instantiate_picture,
)


ROOT = Path(__file__).resolve().parents[1]


def _source(options: str = "view=spatial") -> str:
    return rf"""
\begin{{tikzpicture}}[3d view={{38}}{{24}}]
  \coordinate (A) at (0,0,0);
  \coordinate (Z) at (0,0,1);
  \coordinate (R) at (1,0,0);
  \coordinate (O) at (0,0,2);
  \coordinate (U) at (0,1,2);
  \coordinate (V) at (-0.8,0,2.6);
  \DeclareSpacePlane{{cut}}{{O/U/V}}
  \DeclareSpaceRightCone{{cone}}{{A/Z/R}}{{30}}{{0/9}}{{open_single}}
  \DeclareDandelinConstruction{{dan}}{{cone}}{{cut}}
  \DrawDandelinDiagram[{options}]{{dan}}
\end{{tikzpicture}}
"""


class TikzNativeDandelinCompilerTests(unittest.TestCase):
    def test_checked_in_example_contains_the_three_distinct_views(self) -> None:
        source = ROOT / "examples" / "tikz_dandelin_views" / "tikz_dandelin_views.tex"
        document = compile_document(source)

        self.assertEqual(len(document.pictures), 3)
        self.assertEqual(
            tuple(
                picture.objects[0].geometry["view"]
                for picture in document.pictures
            ),
            ("spatial", "meridian", "section-plane"),
        )
        self.assertEqual(
            tuple(
                picture.objects[0].geometry["mode"]
                for picture in document.pictures
            ),
            (
                "depth_aware_teaching_transparent",
                "diagrammatic",
                "diagrammatic",
            ),
        )
        self.assertTrue(all(not picture.unsupported for picture in document.pictures))

    def test_three_static_views_share_one_authoritative_construction_shape(self) -> None:
        pictures = {
            view: compile_document(source_text=_source(f"view={view}")).pictures[0]
            for view in ("spatial", "meridian", "section-plane")
        }

        for view, picture in pictures.items():
            with self.subTest(view=view):
                self.assertFalse(picture.unsupported)
                self.assertEqual(set(picture.space_right_cones_3d), {"cone"})
                self.assertEqual(set(picture.dandelin_constructions_3d), {"dan"})
                self.assertEqual(
                    set(picture.dandelin_diagrams),
                    {f"dan:view:{view}"},
                )
                self.assertEqual(len(picture.objects), 1)
                item = picture.objects[0]
                self.assertEqual(item.id, f"dan:view:{view}")
                self.assertEqual(item.kind, "dandelin_diagram")
                self.assertEqual(item.geometry["view"], view)
                self.assertEqual(item.geometry["mode"], "diagrammatic")
                self.assertIs(item.geometry["visibilityAuthoritative"], False)
                self.assertIs(item.geometry["static"], True)
                semantic = item.geometry["semanticObjects"]
                self.assertEqual(
                    len({entry["id"] for entry in semantic}),
                    len(semantic),
                )
                self.assertTrue(all(entry["sourceRef"] for entry in semantic))

        meridian_sources = {
            item["sourceRef"]
            for item in pictures["meridian"].objects[0].geometry[
                "semanticObjects"
            ]
            if item["role"] == "focus"
        }
        section_sources = {
            item["sourceRef"]
            for item in pictures["section-plane"].objects[0].geometry[
                "semanticObjects"
            ]
            if item["role"] == "focus"
        }
        self.assertEqual(meridian_sources, section_sources)

    def test_registries_survive_document_serialization(self) -> None:
        document = compile_document(source_text=_source())
        payload = document.to_dict()["pictures"][0]

        self.assertIn("cone", payload["space_right_cones_3d"])
        self.assertIn("dan", payload["dandelin_constructions_3d"])
        self.assertIn("dan:view:spatial", payload["dandelin_diagrams"])
        layers = {
            layer.name: layer.object_ids
            for layer in semantic_animation_layers(document.pictures[0])
        }
        self.assertEqual(layers["solid_geometry"], ("dan:view:spatial",))

    def test_section_plane_rejects_an_invented_sphere_contact_circle(self) -> None:
        picture = compile_document(
            source_text=_source(
                "view=section-plane,show-contact-circles=true"
            )
        ).pictures[0]

        self.assertTrue(picture.unsupported)
        self.assertFalse(picture.dandelin_diagrams)
        self.assertFalse(picture.objects)
        self.assertIn("section-plane", " ".join(picture.unsupported))

    def test_meridian_directrices_reject_during_compilation(self) -> None:
        picture = compile_document(
            source_text=_source("view=meridian,show-directrices=true")
        ).pictures[0]

        self.assertTrue(picture.unsupported)
        self.assertFalse(picture.dandelin_diagrams)
        self.assertFalse(picture.objects)
        self.assertIn("meridian", " ".join(picture.unsupported))

    def test_meridian_sphere_circles_are_unconditional_and_contact_points_toggle(
        self,
    ) -> None:
        pictures = {
            enabled: compile_document(
                source_text=_source(
                    "view=meridian,show-contact-circles="
                    + ("true" if enabled else "false")
                )
            ).pictures[0]
            for enabled in (False, True)
        }
        for enabled, picture in pictures.items():
            with self.subTest(enabled=enabled):
                roles = [
                    item["role"]
                    for item in picture.objects[0].geometry["semanticObjects"]
                ]
                self.assertEqual(roles.count("sphere_circle_section"), 2)
                self.assertEqual(
                    roles.count("contact_circle_section_point"),
                    4 if enabled else 0,
                )
                runtime = instantiate_picture(picture).objects["dan:view:meridian"]
                runtime_roles = [
                    item.dandelin_metadata["semanticKind"]
                    for item in runtime.submobjects
                ]
                self.assertEqual(runtime_roles.count("sphere_circle_section"), 2)
                self.assertEqual(
                    runtime_roles.count("contact_circle_section_point"),
                    4 if enabled else 0,
                )

    def test_automatic_hidden_lines_compile_and_physical_mode_fails_closed(
        self,
    ) -> None:
        automatic = compile_document(
            source_text=_source(
                "view=spatial,mode=depth_aware_diagrammatic"
            )
        ).pictures[0]
        self.assertFalse(automatic.unsupported)
        self.assertEqual(
            automatic.objects[0].geometry["mode"],
            "depth_aware_diagrammatic",
        )
        self.assertIs(
            automatic.objects[0].geometry["curveVisibilityAuthoritative"],
            True,
        )
        self.assertIs(
            automatic.objects[0].geometry["surfaceVisibilityAuthoritative"],
            False,
        )
        runtime = instantiate_picture(automatic).objects["dan:view:spatial"]
        self.assertEqual(runtime.mode, "depth_aware_diagrammatic")
        self.assertTrue(runtime.curve_visibility_authoritative)
        self.assertGreater(runtime.metadata["hiddenSpanCount"], 0)

        physical = compile_document(
            source_text=_source("view=spatial,mode=physical")
        ).pictures[0]
        self.assertTrue(physical.unsupported)
        self.assertFalse(physical.dandelin_diagrams)

        non_spatial = compile_document(
            source_text=_source(
                "view=meridian,mode=depth_aware_diagrammatic"
            )
        ).pictures[0]
        self.assertTrue(non_spatial.unsupported)
        self.assertFalse(non_spatial.dandelin_diagrams)

        analytic = _source().replace(
            "{open_single}",
            "{analytic_double}",
        ).replace("{0/9}", "{-9/9}")
        rejected = compile_document(source_text=analytic).pictures[0]
        self.assertTrue(rejected.unsupported)
        self.assertFalse(rejected.space_right_cones_3d)
        self.assertFalse(rejected.dandelin_constructions_3d)
        self.assertFalse(rejected.dandelin_diagrams)

    def test_teaching_transparent_mode_instantiates_certified_surface_layers(
        self,
    ) -> None:
        picture = compile_document(
            source_text=_source(
                "view=spatial,mode=depth_aware_teaching_transparent"
            )
        ).pictures[0]
        self.assertFalse(picture.unsupported)
        geometry = picture.objects[0].geometry
        self.assertIs(geometry["curveVisibilityAuthoritative"], True)
        self.assertIs(geometry["surfaceLayeringAuthoritative"], True)
        self.assertIs(geometry["surfaceVisibilityAuthoritative"], False)
        self.assertIs(
            geometry["physicalSurfaceVisibilityAuthoritative"],
            False,
        )

        runtime = instantiate_picture(picture).objects["dan:view:spatial"]
        self.assertEqual(runtime.mode, "depth_aware_teaching_transparent")
        self.assertTrue(runtime.curve_visibility_authoritative)
        self.assertTrue(runtime.surface_layering_authoritative)
        self.assertFalse(runtime.physical_surface_visibility_authoritative)
        surface_ids = set(runtime.surface_layer_frame.draw_order)
        rendered = {
            member.dandelin_metadata["paintItemId"]: member.z_index
            for wrapper in runtime.submobjects
            for member in wrapper.submobjects
            if isinstance(getattr(member, "dandelin_metadata", None), dict)
            and member.dandelin_metadata.get("paintItemId") in surface_ids
        }
        self.assertEqual(set(rendered), surface_ids)
        self.assertEqual(
            tuple(sorted(rendered, key=rendered.__getitem__)),
            runtime.surface_layer_frame.draw_order,
        )

    def test_teaching_transparent_mode_compiles_with_separate_authority(self) -> None:
        picture = compile_document(
            source_text=_source(
                "view=spatial,mode=depth_aware_teaching_transparent"
            )
        ).pictures[0]

        self.assertFalse(picture.unsupported)
        geometry = picture.objects[0].geometry
        self.assertEqual(geometry["mode"], "depth_aware_teaching_transparent")
        self.assertIs(geometry["visibilityAuthoritative"], False)
        self.assertIs(geometry["curveVisibilityAuthoritative"], True)
        self.assertIs(geometry["surfaceVisibilityAuthoritative"], False)
        self.assertIs(geometry["surfaceLayeringAuthoritative"], True)
        self.assertIs(
            geometry["physicalSurfaceVisibilityAuthoritative"],
            False,
        )

        non_spatial = compile_document(
            source_text=_source(
                "view=section-plane,mode=depth_aware_teaching_transparent"
            )
        ).pictures[0]
        self.assertTrue(non_spatial.unsupported)
        self.assertFalse(non_spatial.dandelin_diagrams)

        hidden_seams = compile_document(
            source_text=_source(
                "view=spatial,mode=depth_aware_teaching_transparent,"
                "show-contact-circles=false"
            )
        ).pictures[0]
        self.assertTrue(hidden_seams.unsupported)
        self.assertFalse(hidden_seams.dandelin_diagrams)
        self.assertFalse(hidden_seams.objects)
        self.assertIn("equal-depth seams", " ".join(hidden_seams.unsupported))

        hyperbola = _source(
            "view=spatial,mode=depth_aware_teaching_transparent"
        ).replace(
            r"\coordinate (V) at (-0.8,0,2.6);",
            r"\coordinate (V) at (-0.2,0,2.979795897113271);",
        ).replace(
            r"{30}{0/9}{open_single}",
            r"{30}{-20/20}{open_double}",
        )
        blocked = compile_document(source_text=hyperbola).pictures[0]
        self.assertTrue(blocked.unsupported)
        self.assertFalse(blocked.dandelin_diagrams)
        self.assertFalse(blocked.objects)

    def test_invalid_declaration_does_not_leave_a_partial_registry_entry(self) -> None:
        source = r"""
\begin{tikzpicture}[3d view={38}{24}]
  \coordinate (A) at (0,0,0);
  \coordinate (Z) at (0,0,1);
  \coordinate (R) at (0,0,2);
  \DeclareSpaceRightCone{bad-cone}{A/Z/R}{30}{0/9}{open_single}
\end{tikzpicture}
"""
        picture = compile_document(source_text=source).pictures[0]

        self.assertTrue(picture.unsupported)
        self.assertFalse(picture.space_right_cones_3d)

    def test_diagram_cannot_share_a_picture_with_ordinary_drawables(self) -> None:
        mixed = _source().replace(
            r"\DrawDandelinDiagram[view=spatial]{dan}",
            (
                r"\draw (A) -- (Z);"
                "\n"
                r"\DrawDandelinDiagram[view=spatial]{dan}"
            ),
        )
        picture = compile_document(source_text=mixed).pictures[0]

        self.assertTrue(picture.unsupported)
        self.assertFalse(picture.dandelin_diagrams)
        self.assertEqual([item.kind for item in picture.objects], ["line"])

    def test_fixed_renderer_instantiates_all_three_certified_views(self) -> None:
        for view in ("spatial", "meridian", "section-plane"):
            with self.subTest(view=view):
                picture = compile_document(
                    source_text=_source(f"view={view}")
                ).pictures[0]
                figure = instantiate_picture(picture, scene_unit_per_cm=0.5)
                group = figure.objects[f"dan:view:{view}"]

                self.assertGreater(group.width, 0.0)
                self.assertGreater(group.height, 0.0)
                self.assertEqual(group.dandelin_metadata["view"], view)
                self.assertEqual(group.dandelin_metadata["mode"], "diagrammatic")
                self.assertIs(
                    group.dandelin_metadata["visibilityAuthoritative"],
                    False,
                )
                kinds = {
                    item.dandelin_metadata["semanticKind"]
                    for item in group.submobjects
                }
                if view == "section-plane":
                    self.assertTrue(
                        kinds.isdisjoint(
                            {
                                "sphere_surface",
                                "sphere_circle_section",
                                "contact_circle",
                            }
                        )
                    )
                    self.assertIn("section_curve", kinds)
                    self.assertIn("focus", kinds)

    def test_provider_preserves_depth_aware_fragment_painter_graph(self) -> None:
        picture = compile_document(
            source_text=_source(
                "view=spatial,mode=depth_aware_diagrammatic"
            )
        ).pictures[0]
        spec = picture.objects[0]
        runtime = instantiate_picture(picture).objects[spec.id]

        self.assertEqual(runtime.z_index, spec.z_index)
        family = runtime.get_family()[1:]

        def painter_z_indices(**expected: str) -> list[float]:
            return [
                float(member.z_index)
                for member in family
                if all(
                    getattr(member, "dandelin_metadata", {}).get(key) == value
                    for key, value in expected.items()
                )
            ]

        hidden = painter_z_indices(
            visibilityKind="hidden",
            renderIntent="dashed",
        )
        visible = painter_z_indices(
            visibilityKind="visible",
            renderIntent="solid",
        )
        foci = painter_z_indices(semanticKind="focus")

        self.assertTrue(hidden)
        self.assertTrue(visible)
        self.assertTrue(foci)
        z_by_item = {
            member.dandelin_metadata["painterItemId"]: float(member.z_index)
            for member in family
            if isinstance(getattr(member, "dandelin_metadata", None), dict)
            and isinstance(
                member.dandelin_metadata.get("painterItemId"),
                str,
            )
        }
        self.assertEqual(
            set(z_by_item),
            set(runtime.compositing_frame.draw_order),
        )
        self.assertEqual(
            tuple(sorted(z_by_item, key=z_by_item.__getitem__)),
            runtime.compositing_frame.draw_order,
        )
        self.assertEqual(len(set(z_by_item.values())), len(z_by_item))
        self.assertLess(max(z_by_item.values()), min(foci))

    def test_provider_persists_nested_semantic_source_refs(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "dandelin.tex"
            source.write_text(_source("view=spatial"), encoding="utf-8")
            compiled = compile_asset(source)

        self.assertEqual(compiled.selected_compatibility["overall_level"], "B")
        self.assertEqual(compiled.selected_compatibility["static_status"], "pass")
        self.assertGreater(compiled.asset["bounds"]["width_scene"], 0.0)
        self.assertGreater(compiled.asset["bounds"]["height_scene"], 0.0)
        index = compiled.asset["object_index"]
        self.assertEqual(len(index), 1)
        self.assertEqual(index[0]["id"], "dan:view:spatial")
        self.assertEqual(index[0]["kind"], "dandelin_diagram")
        self.assertTrue(index[0]["semantic_objects"])
        self.assertTrue(
            all(item["sourceRef"] for item in index[0]["semantic_objects"])
        )
        runtime = compiled.figure.objects["dan:view:spatial"]
        runtime_index = [
            {
                "id": item.dandelin_metadata["semanticId"],
                "role": item.dandelin_metadata["semanticKind"],
                "sourceRef": item.dandelin_metadata["semanticSourceRefs"][0],
            }
            for item in runtime.submobjects
        ]
        self.assertEqual(index[0]["semantic_objects"], runtime_index)
        self.assertEqual(
            len(runtime_index),
            len({item["id"] for item in runtime_index}),
        )

    def test_provider_semantic_index_matches_runtime_bidirectionally_for_all_views(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "dandelin.tex"
            for view in ("spatial", "meridian", "section-plane"):
                with self.subTest(view=view):
                    source.write_text(
                        _source(f"view={view}"),
                        encoding="utf-8",
                    )
                    compiled = compile_asset(source)
                    advertised = compiled.asset["object_index"][0][
                        "semantic_objects"
                    ]
                    runtime = compiled.figure.objects[f"dan:view:{view}"]
                    actual = [
                        {
                            "id": item.dandelin_metadata["semanticId"],
                            "role": item.dandelin_metadata["semanticKind"],
                            "sourceRef": item.dandelin_metadata[
                                "semanticSourceRefs"
                            ][0],
                        }
                        for item in runtime.submobjects
                    ]
                    self.assertEqual(advertised, actual)
                    self.assertEqual(len(advertised), len(runtime.submobjects))

    def test_named_plane_coordinate_changes_fail_before_instantiation(self) -> None:
        changes = {
            "O": (0.0, 0.0, 3.0),
            "U": (0.0, 2.0, 2.0),
            "V": (-1.2, 0.0, 2.6),
        }
        for name, value in changes.items():
            with self.subTest(name=name):
                picture = compile_document(source_text=_source()).pictures[0]
                picture.coordinates[name] = value
                with self.assertRaisesRegex(
                    TikzNativeProviderError,
                    "point evidence|named coordinates",
                ):
                    instantiate_picture(picture)


if __name__ == "__main__":
    unittest.main()
