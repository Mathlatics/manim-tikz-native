from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

from manim import Scene, ValueTracker, tempconfig

from tikz_native.compiler import compile_document
from tikz_native.geometry_rig_3d import analyze_geometry_rig_3d
from tikz_native.native_manim_codegen_3d_v3 import (
    generate_native_manim_source_3d_v3,
)
from tikz_native.open_face_static_asset_3d import (
    OPEN_FACE_STATIC_ENTRY_3D_SCHEMA,
    TikzNativeOpenFaceStaticAsset3DError,
    bake_open_face_static_entry_3d,
)
from tikz_native.provider import instantiate_picture
from tikz_native.version import (
    provider_component_contract_revisions,
    provider_component_revisions,
    provider_revision,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "dihedral_fold_3d_demo" / "dihedral_fold.tex"


class TikzNativeOpenFaceStaticAsset3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_text = SOURCE.read_text(encoding="utf-8")
        cls.source_sha = hashlib.sha256(cls.source_text.encode("utf-8")).hexdigest()
        cls.picture = compile_document(source_text=cls.source_text).pictures[0]
        cls.rig = analyze_geometry_rig_3d(cls.picture)
        cls.source_v3 = generate_native_manim_source_3d_v3(cls.picture, cls.rig)

    def setUp(self) -> None:
        self.config = tempconfig(
            {"renderer": "cairo", "pixel_width": 320, "pixel_height": 180}
        )
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def _contract(self) -> dict:
        visibility = self.source_v3["visibilitySpec"]
        components = provider_component_revisions()
        contracts = provider_component_contract_revisions()
        return {
            "schema": "latex-ppt-tikz-native-open-face-static-asset/v2",
            "mode": "open_convex_faces_parallel",
            "sourceSha256": self.source_sha,
            "pictureIndex": 1,
            "entryMacro": "",
            "modelSha256": visibility["modelSha256"],
            "entryTraceSha256": visibility["entryTraceSha256"],
            "adapterResultSha256": visibility["adapterResultSha256"],
            "faceCount": visibility["faceCount"],
            "strokeCount": visibility["strokeCount"],
            "seamCount": visibility["seamCount"],
            "compatibility": {
                "schema": "tikz-native-artifact-compatibility/v1",
                "contractRevisions": {
                    name: contracts[name]
                    for name in (
                        "tikz_open_face_static_asset_3d",
                        "native_manim_source_3d_v3",
                    )
                },
                "renderRevisions": {
                    name: components[name]
                    for name in (
                        "tikz_open_face_static_asset_3d",
                        "native_manim_source_3d_v3",
                    )
                },
                "buildRevision": provider_revision(),
            },
        }

    def _legacy_contract(self) -> dict:
        visibility = self.source_v3["visibilitySpec"]
        components = provider_component_revisions()
        return {
            "schema": "latex-ppt-tikz-native-open-face-static-asset/v1",
            "mode": "open_convex_faces_parallel",
            "sourceSha256": self.source_sha,
            "pictureIndex": 1,
            "entryMacro": "",
            "modelSha256": visibility["modelSha256"],
            "entryTraceSha256": visibility["entryTraceSha256"],
            "adapterResultSha256": visibility["adapterResultSha256"],
            "faceCount": visibility["faceCount"],
            "strokeCount": visibility["strokeCount"],
            "seamCount": visibility["seamCount"],
            "componentRevisions": {
                name: components[name]
                for name in (
                    "asset_compiler",
                    "open_face_visibility",
                    "tikz_open_face_visibility_3d",
                    "tikz_open_face_static_asset_3d",
                    "native_manim_source_3d_v3",
                )
            },
        }

    def _bake(self):
        figure = instantiate_picture(self.picture, scene_unit_per_cm=1.0)
        bake_open_face_static_entry_3d(
            figure,
            self._contract(),
            source_sha256=self.source_sha,
            picture_index=1,
        )
        return figure

    def test_bake_preserves_semantic_objects_and_adds_one_frozen_overlay(self) -> None:
        figure = instantiate_picture(self.picture, scene_unit_per_cm=1.0)
        object_identity = {key: id(value) for key, value in figure.objects.items()}
        children = tuple(id(value) for value in figure.group.submobjects)

        bake_open_face_static_entry_3d(
            figure,
            self._contract(),
            source_sha256=self.source_sha,
            picture_index=1,
        )

        self.assertEqual(
            object_identity,
            {key: id(value) for key, value in figure.objects.items()},
        )
        self.assertEqual(
            children,
            tuple(id(value) for value in figure.group.submobjects[:-1]),
        )
        record = figure.group._mathppt_open_face_static_entry
        self.assertEqual(record["schema"], OPEN_FACE_STATIC_ENTRY_3D_SCHEMA)
        self.assertGreater(record["strokeWidthPerPt"], 1.0)
        self.assertIs(record["overlayRoot"], figure.group.submobjects[-1])
        self.assertEqual(record["entryTraceSha256"], self._contract()["entryTraceSha256"])
        element_objects = figure.group._codex_tikz_native_element_objects
        self.assertEqual(set(element_objects), set(figure.objects))
        self.assertTrue(
            any(element_objects[key] is not figure.objects[key] for key in figure.objects)
        )

    def test_stale_contract_fails_transactionally(self) -> None:
        figure = instantiate_picture(self.picture, scene_unit_per_cm=1.0)
        children = tuple(id(value) for value in figure.group.submobjects)
        styles = {
            key: tuple(
                float(member.get_stroke_opacity())
                for member in value.get_family()
                if hasattr(member, "get_stroke_opacity")
            )
            for key, value in figure.objects.items()
        }
        contract = self._contract()
        contract["entryTraceSha256"] = "0" * 64
        with self.assertRaises(TikzNativeOpenFaceStaticAsset3DError):
            bake_open_face_static_entry_3d(
                figure,
                contract,
                source_sha256=self.source_sha,
                picture_index=1,
            )
        self.assertEqual(children, tuple(id(value) for value in figure.group.submobjects))
        self.assertFalse(hasattr(figure.group, "_mathppt_open_face_static_entry"))
        self.assertEqual(
            styles,
            {
                key: tuple(
                    float(member.get_stroke_opacity())
                    for member in value.get_family()
                    if hasattr(member, "get_stroke_opacity")
                )
                for key, value in figure.objects.items()
            },
        )

    def test_v2_render_and_build_drift_do_not_invalidate_author_data(self) -> None:
        contract = self._contract()
        compatibility = contract["compatibility"]
        compatibility["renderRevisions"] = {
            name: "source-sha256:" + "1" * 64
            for name in compatibility["renderRevisions"]
        }
        compatibility["buildRevision"] = "source-sha256:" + "2" * 64
        figure = instantiate_picture(self.picture, scene_unit_per_cm=1.0)
        bake_open_face_static_entry_3d(
            figure,
            contract,
            source_sha256=self.source_sha,
            picture_index=1,
        )
        self.assertEqual(
            figure.group._mathppt_open_face_static_entry["contractSchema"],
            "latex-ppt-tikz-native-open-face-static-asset/v2",
        )

    def test_v2_contract_drift_remains_fail_closed(self) -> None:
        contract = self._contract()
        contract["compatibility"]["contractRevisions"][
            "native_manim_source_3d_v3"
        ] = "tikz-native-contract:native_manim_source_3d_v3/v999"
        figure = instantiate_picture(self.picture, scene_unit_per_cm=1.0)
        with self.assertRaisesRegex(
            TikzNativeOpenFaceStaticAsset3DError, "incompatible"
        ):
            bake_open_face_static_entry_3d(
                figure,
                contract,
                source_sha256=self.source_sha,
                picture_index=1,
            )

    def test_legacy_v1_retains_exact_render_revision_gate(self) -> None:
        contract = self._legacy_contract()
        contract["componentRevisions"]["native_manim_source_3d_v3"] = (
            "source-sha256:" + "3" * 64
        )
        figure = instantiate_picture(self.picture, scene_unit_per_cm=1.0)
        with self.assertRaisesRegex(
            TikzNativeOpenFaceStaticAsset3DError, "stale"
        ):
            bake_open_face_static_entry_3d(
                figure,
                contract,
                source_sha256=self.source_sha,
                picture_index=1,
            )

    def test_v3_dynamic_overlay_detaches_and_restores_baked_entry(self) -> None:
        namespace: dict[str, object] = {}
        exec(
            compile(self.source_v3["sourceText"], "<native-source-v3>", "exec"),
            namespace,
        )
        figure = self._bake()
        shape = figure.group.scale(0.72).rotate(0.13).shift((1.1, -0.45, 0.0))
        entry_children = tuple(id(item) for item in shape.submobjects)
        baked_overlay = shape._mathppt_open_face_static_entry["overlayRoot"]
        scene = Scene()
        scene.add(shape)
        trackers = {
            driver_id: ValueTracker(initial)
            for driver_id, initial in namespace["DRIVER_INITIAL_VALUES"].items()
        }
        geometry = namespace["install_geometry_3d_updaters"](
            shape,
            figure.objects,
            trackers,
            ValueTracker(namespace["CAMERA_PROGRESS_INITIAL"]),
        )
        self.assertAlmostEqual(
            geometry["stroke_width_per_pt"],
            shape._mathppt_open_face_static_entry["strokeWidthPerPt"],
            places=12,
        )
        visibility = namespace["install_open_face_visibility_3d"](
            scene, shape, figure.objects, geometry
        )
        self.assertNotIn(baked_overlay, shape.submobjects)
        self.assertIn(visibility["overlay_root"], scene.mobjects)
        active_widths = [
            float(line.get_stroke_width())
            for line in visibility["overlay_root"].get_family()
            if callable(getattr(line, "has_points", None))
            and line.has_points()
            and float(line.get_stroke_opacity()) > 0.0
        ]
        self.assertTrue(active_widths)
        self.assertGreater(min(active_widths), 1.0)

        driver_id = next(iter(trackers))
        driver_spec = namespace["DRIVER_SPECS"][driver_id]
        trackers[driver_id].set_value(
            0.5 * sum(float(value) for value in driver_spec["range"])
        )
        shape.update(0.0)
        visibility["overlay_root"].update(0.0)
        moving_widths = [
            float(line.get_stroke_width())
            for line in visibility["overlay_root"].get_family()
            if callable(getattr(line, "has_points", None))
            and line.has_points()
            and float(line.get_stroke_opacity()) > 0.0
        ]
        self.assertTrue(moving_widths)
        self.assertGreater(min(moving_widths), 1.0)

        namespace["restore_open_face_visibility_3d"](visibility)
        namespace["restore_geometry_3d_objects"](geometry)
        self.assertEqual(entry_children, tuple(id(item) for item in shape.submobjects))
        self.assertIn(baked_overlay, shape.submobjects)
        self.assertNotIn(visibility["overlay_root"], scene.mobjects)


if __name__ == "__main__":
    unittest.main()
