from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator

from tikz_native.compatibility import audit_document_compatibility
from tikz_native.compiler import compile_document
from tikz_native.geometry_rig_3d import (
    GEOMETRY_RIG_3D_SCHEMA,
    HINGE_FOLD_CANDIDATE_ID_PREFIX,
    analyze_geometry_rig_3d,
    attach_geometry_rig_3d_identity,
    semantic_model_3d_hash,
)
from tikz_native.motion_3d import Motion3DSpec
from tikz_native.motion_3d_bridge import provider_info, sha256_file
from tikz_native.native_manim_codegen_3d import NativeManimCodegen3DError
from tikz_native.native_manim_codegen_3d_v2 import NativeManimCodegen3DV2Error


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "dihedral_fold_3d_demo"
SOURCE = DEMO / "dihedral_fold.tex"
SCHEMA = ROOT / "tikz_native" / "schemas" / "geometry-rig-3d-v1.schema.json"


class TikzNativeGeometryRig3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.picture = compile_document(SOURCE).pictures[0]
        cls.schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)

    @staticmethod
    def _selection() -> dict:
        return {
            "candidate_id": f"{HINGE_FOLD_CANDIDATE_ID_PREFIX}fold-angle",
            "range": [0.3141592653589793, 1.9547687622336491],
        }

    def _portable(self, rig: dict, *, expected_revision: str | None = None) -> dict:
        revision = str(provider_info()["revision"])
        return attach_geometry_rig_3d_identity(
            rig,
            source_sha256=sha256_file(SOURCE),
            provider_revision=revision,
            expected_asset_provider_revision=expected_revision or revision,
        )

    def test_fixture_exposes_explicit_hinge_projection_and_polygon_point_names(self) -> None:
        picture = self.picture
        self.assertEqual(picture.dimension, 3)
        self.assertEqual(len(picture.hinge_relations), 1)
        hinge = picture.hinge_relations[0]
        self.assertEqual(hinge.schema, "tikz-native-hinge-relation/v1")
        self.assertEqual(hinge.id, "fold-angle")
        self.assertEqual(hinge.axis_names, ["A", "B"])
        self.assertEqual(
            hinge.fixed_face_names,
            ["A", "B", "Alpha1", "Alpha0"],
        )
        self.assertEqual(
            hinge.moving_face_names,
            ["A", "B", "Beta1", "Beta0"],
        )
        self.assertEqual(
            picture.coordinate_dependencies["N"],
            {
                "operation": "projection",
                "line_start": "A",
                "point": "M",
                "line_end": "B",
            },
        )
        polygons = {item.id: item for item in picture.objects if item.kind == "polygon"}
        self.assertEqual(
            polygons["fill.A.B.Alpha1.Alpha0"].geometry["point_names"],
            ["A", "B", "Alpha1", "Alpha0"],
        )
        self.assertEqual(
            polygons[
                "plane_interaction_fill.A.B.Beta1.Beta0"
            ].geometry["point_names"],
            ["A", "B", "Beta1", "Beta0"],
        )
        dots = {item.id: item for item in picture.objects if item.kind == "dot"}
        self.assertEqual(dots["dot.M"].geometry["center_name"], "M")
        serialized = compile_document(SOURCE).to_dict()["pictures"][0]
        self.assertEqual(
            serialized["hinge_relations"][0]["schema"],
            "tikz-native-hinge-relation/v1",
        )

    def test_explicit_hinge_is_a_registered_dynamic_safe_subset_feature(self) -> None:
        report = audit_document_compatibility(compile_document(SOURCE))
        features = {item["id"]: item for item in report["encountered_features"]}
        self.assertEqual(features["relation.hinge_3d"]["level"], "A")
        self.assertEqual(features["relation.hinge_3d"]["count"], 1)

    def test_degenerate_hinge_face_is_rejected_by_the_compiler(self) -> None:
        source = r"""
\begin{tikzpicture}[space view={(-0.35,-0.35),(1,0),(0,1)}]
  \coordinate (A) at (0,0,0);
  \coordinate (B) at (2,0,0);
  \coordinate (C) at (1,0,0);
  \coordinate (D) at (0,1,0);
  \DeclareSpaceHinge{bad-fold}{A/B}{A/B/C}{A/B/D}
\end{tikzpicture}
"""
        picture = compile_document(source_text=source).pictures[0]
        self.assertFalse(picture.hinge_relations)
        self.assertTrue(
            any(
                "fixed face needs a point outside the axis" in finding
                for finding in picture.unsupported
            ),
            picture.unsupported,
        )

    def test_discovery_classifies_hinge_relations_and_camera_without_compiling_motion(self) -> None:
        rig = analyze_geometry_rig_3d(self.picture)
        self.assertEqual(rig["schema"], GEOMETRY_RIG_3D_SCHEMA)
        self.assertEqual(rig["dimension"], 3)
        self.assertEqual(rig["status"], "needs_selection")
        self.assertIsNone(rig["motionSpecCore"])
        self.assertIsNone(rig["nativeManimSource"])
        self.assertEqual(
            rig["nativeManimSourceV2"]["schema"],
            "tikz-native-manim-source-3d/v2",
        )

        candidates = rig["motionCandidates"]
        by_kind: dict[str, list[dict]] = {}
        for item in candidates:
            by_kind.setdefault(item["candidateKind"], []).append(item)
        geometry_drivers = {
            item["driverType"]: item for item in by_kind["geometry_driver"]
        }
        self.assertEqual(set(geometry_drivers), {"hinge_fold", "point_on_segment"})
        hinge = geometry_drivers["hinge_fold"]
        self.assertEqual(hinge["driverType"], "hinge_fold")
        self.assertEqual(hinge["relationId"], "fold-angle")
        self.assertEqual(hinge["axis"], ["A", "B"])
        self.assertEqual(hinge["movingCoordinates"], ["Beta1", "Beta0"])
        self.assertAlmostEqual(hinge["initial"]["value"], 1.0122909662, places=8)
        self.assertLess(
            hinge["suggestedRange"]["minimum"], hinge["initial"]["value"]
        )
        self.assertGreater(
            hinge["suggestedRange"]["maximum"], hinge["initial"]["value"]
        )
        point = geometry_drivers["point_on_segment"]
        self.assertEqual(point["driverId"], "point_on_segment:M")
        self.assertEqual(point["coordinateId"], "M")
        self.assertEqual(point["segment"], ["Beta0", "Beta1"])
        self.assertEqual(point["initial"], {"value": 0.67, "unit": "ratio"})
        cameras = {
            item["mode"]: item
            for item in by_kind["camera_operation"]
        }
        self.assertEqual(set(cameras), {"front", "side", "top", "oblique", "isometric"})
        # This fixture's authored TikZ basis is a proper rotation frame.  Its
        # orthogonal presets can orbit, while the general oblique projection
        # must keep the legacy runtime on linear interpolation.
        self.assertEqual(cameras["oblique"]["transitionTypes"], ["linear"])
        for mode in ("front", "side", "top", "isometric"):
            self.assertEqual(
                cameras[mode]["transitionTypes"],
                ["linear", "orbit"],
            )

        derived = {
            item["coordinateId"]: item for item in by_kind["derived_relation"]
        }
        self.assertEqual(derived["M"]["relationType"], "point_on_segment")
        self.assertEqual(derived["M"]["dependsOn"], ["Beta0", "Beta1"])
        self.assertTrue(derived["M"]["affectedByDriver"])
        self.assertEqual(derived["N"]["relationType"], "project_point_to_line")
        self.assertEqual(derived["N"]["dependsOn"], ["M", "A", "B"])
        self.assertTrue(derived["N"]["affectedByDriver"])

        camera_modes = {
            item["mode"] for item in by_kind["camera_operation"]
        }
        self.assertEqual(
            camera_modes,
            {"front", "side", "top", "oblique", "isometric"},
        )
        self.assertTrue(
            all(item["restoresEntry"] for item in by_kind["camera_operation"])
        )

    def test_confirmed_hinge_builds_current_motion_3d_v1_core_and_roles(self) -> None:
        rig = analyze_geometry_rig_3d(self.picture, selection=self._selection())
        self.assertEqual(rig["status"], "ready")
        self.assertEqual(
            rig["selectedMotionCandidate"]["candidateId"],
            f"{HINGE_FOLD_CANDIDATE_ID_PREFIX}fold-angle",
        )
        self.assertEqual(
            rig["affectedCoordinateIds"], ["Beta0", "Beta1", "M", "N"]
        )
        self.assertEqual(
            rig["fixedCoordinateIds"],
            ["A", "B", "Alpha0", "Alpha1", "S", "E"],
        )
        roles = {item["coordinateId"]: item for item in rig["coordinateRoles"]}
        self.assertEqual(roles["A"]["role"], "hinge_axis")
        self.assertEqual(roles["Beta0"]["role"], "driver_coordinate")
        self.assertEqual(roles["M"]["role"], "derived")
        self.assertEqual(roles["Alpha0"]["role"], "fixed")

        groups = {item["groupId"]: item for item in rig["semanticGroups"]}
        self.assertEqual(len(groups), len(rig["semanticGroups"]))
        self.assertEqual(
            set(groups),
            {
                "hinge:fold-angle:fixed-face",
                "hinge:fold-angle:moving-face",
                "hinge:fold-angle:axis",
                "derived:M",
                "derived:N",
                *(
                    f"occlusion:{relation.id}"
                    for relation in self.picture.occlusion_relations
                ),
            },
        )
        self.assertEqual(
            groups["hinge:fold-angle:fixed-face"]["coordinateIds"],
            ["A", "B", "Alpha1", "Alpha0"],
        )
        self.assertIn(
            "fill.A.B.Alpha1.Alpha0",
            groups["hinge:fold-angle:fixed-face"]["objectIds"],
        )
        self.assertIn(
            "occluded_visible.Alpha1.Alpha0.0",
            groups["hinge:fold-angle:fixed-face"]["objectIds"],
        )
        self.assertEqual(
            groups["hinge:fold-angle:moving-face"]["coordinateIds"],
            ["A", "B", "Beta1", "Beta0"],
        )
        self.assertIn(
            "plane_interaction_fill.A.B.Beta1.Beta0",
            groups["hinge:fold-angle:moving-face"]["objectIds"],
        )
        self.assertIn(
            "occluded_visible.Beta1.Beta0.0",
            groups["hinge:fold-angle:moving-face"]["objectIds"],
        )
        self.assertEqual(
            groups["hinge:fold-angle:axis"]["objectIds"],
            ["occluded_visible.A.B.0", "occluded_visible.A.B.0.2"],
        )
        self.assertEqual(
            groups["derived:M"]["objectIds"],
            ["line.M.N", "dot.M", "label.M.M"],
        )
        self.assertEqual(
            groups["derived:N"]["objectIds"],
            ["line.M.N", "dot.N", "label.N.N"],
        )
        probe = groups[
            "occlusion:occlusion_relation.S.E.A.B.Beta1.Beta0"
        ]
        self.assertEqual(probe["roles"], ["probe", "occlusion", "follower"])
        self.assertFalse(probe["required"])
        raw_fragment_ids = {
            object_id
            for relation in self.picture.occlusion_relations
            for object_id in relation.object_ids
        }
        grouped_fragment_ids = {
            object_id
            for group in rig["semanticGroups"]
            if "occlusion" in group["roles"]
            for object_id in group["objectIds"]
        }
        self.assertEqual(grouped_fragment_ids, raw_fragment_ids)

        bindings = {item["objectId"]: item for item in rig["bindings"]}
        self.assertEqual(
            bindings["plane_interaction_fill.A.B.Beta1.Beta0"]["role"],
            "active",
        )
        self.assertEqual(bindings["line.M.N"]["role"], "follower")
        self.assertEqual(bindings["fill.A.B.Alpha1.Alpha0"]["role"], "fixed")
        self.assertFalse(
            any(item["objectId"].startswith("occluded_") for item in rig["bindings"])
        )

        occlusions = rig["occlusionBindings"]
        self.assertEqual(len(occlusions), 9)
        self.assertEqual(
            sum(bool(item["dynamicByGeometry"]) for item in occlusions),
            8,
        )
        probe = next(
            item
            for item in occlusions
            if item["linePointNames"] == ["S", "E"]
        )
        self.assertTrue(probe["dynamicByGeometry"])
        self.assertTrue(probe["cameraSensitive"])
        self.assertEqual(len(probe["objectIds"]), 3)

        core = rig["motionSpecCore"]
        self.assertEqual(core["end_policy"], "restore_entry")
        self.assertEqual(core["driver"]["type"], "hinge_fold")
        self.assertEqual(core["driver"]["moving_points"], ["Beta1", "Beta0"])
        self.assertEqual(core["camera"]["restore_transition"], "linear")
        self.assertEqual(
            [item["name"] for item in core["derived_coordinates"]],
            ["M", "N"],
        )
        payload = {
            "schema": "tikz-native-motion-3d/v1",
            "picture_index": self.picture.index,
            **deepcopy(core),
            "timeline": [{"type": "wait", "duration": 0.1}],
        }
        motion = Motion3DSpec.from_dict(payload)
        motion.validate_picture(self.picture)
        source = rig["nativeManimSource"]
        self.assertEqual(source["schema"], "tikz-native-manim-source-3d/v1")
        self.assertIn("def install_geometry_3d_updaters(", source["sourceText"])
        source_v2 = rig["nativeManimSourceV2"]
        self.assertEqual(source_v2["schema"], "tikz-native-manim-source-3d/v2")
        self.assertEqual(
            source_v2["authoringSpec"]["endPolicy"],
            "restore_entry",
        )

    def test_nonorthogonal_tikz_entry_advertises_no_orbit_transition(self) -> None:
        picture = deepcopy(self.picture)
        picture.projection_3d.matrix = (
            (1.0, 0.25, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        rig = analyze_geometry_rig_3d(picture)
        cameras = [
            item
            for item in rig["motionCandidates"]
            if item["candidateKind"] == "camera_operation"
        ]
        self.assertEqual(len(cameras), 5)
        for camera in cameras:
            self.assertEqual(camera["transitionTypes"], ["linear"])

    def test_codegen_gap_does_not_revoke_the_legacy_ready_rig(self) -> None:
        with patch(
            "tikz_native.geometry_rig_3d.generate_native_manim_source_3d",
            side_effect=NativeManimCodegen3DError("unsupported readable source probe"),
        ):
            rig = analyze_geometry_rig_3d(
                self.picture,
                selection=self._selection(),
            )
        self.assertEqual(rig["status"], "ready")
        self.assertIsNotNone(rig["motionSpecCore"])
        self.assertIsNone(rig["nativeManimSource"])
        diagnostic = next(
            item
            for item in rig["diagnostics"]
            if item["code"] == "NATIVE_MANIM_SOURCE_UNAVAILABLE"
        )
        self.assertEqual(diagnostic["severity"], "warning")

    def test_v2_codegen_gap_does_not_revoke_v1_or_the_legacy_ready_rig(self) -> None:
        with patch(
            "tikz_native.geometry_rig_3d.generate_native_manim_source_3d_v2",
            side_effect=NativeManimCodegen3DV2Error(
                "unsupported multi-driver source probe"
            ),
        ):
            rig = analyze_geometry_rig_3d(
                self.picture,
                selection=self._selection(),
            )
        self.assertEqual(rig["status"], "ready")
        self.assertIsNotNone(rig["motionSpecCore"])
        self.assertIsNotNone(rig["nativeManimSource"])
        self.assertIsNone(rig["nativeManimSourceV2"])
        diagnostic = next(
            item
            for item in rig["diagnostics"]
            if item["code"] == "NATIVE_MANIM_SOURCE_V2_UNAVAILABLE"
        )
        self.assertEqual(diagnostic["severity"], "warning")

    def test_analysis_never_recovers_polygon_points_from_object_id(self) -> None:
        picture = deepcopy(self.picture)
        moving = next(
            item
            for item in picture.objects
            if item.id == "plane_interaction_fill.A.B.Beta1.Beta0"
        )
        moving.geometry.pop("point_names")
        rig = analyze_geometry_rig_3d(picture, selection=self._selection())
        self.assertEqual(rig["status"], "blocked")
        self.assertIsNone(rig["motionSpecCore"])
        self.assertIsNone(rig["nativeManimSource"])
        self.assertIsNone(rig["nativeManimSourceV2"])
        self.assertIn(
            "MOVING_FACE_OBJECT_NOT_FOUND",
            {item["code"] for item in rig["diagnostics"]},
        )

    def test_semantic_hash_includes_hinge_projection_occlusion_and_projection_matrix(self) -> None:
        baseline = semantic_model_3d_hash(self.picture)
        from_text = semantic_model_3d_hash(
            compile_document(source_text=SOURCE.read_text(encoding="utf-8")).pictures[0]
        )
        self.assertEqual(baseline, from_text)
        cases = []
        changed = deepcopy(self.picture)
        changed.hinge_relations[0].moving_face_names[-1] = "Alpha0"
        cases.append(changed)
        changed = deepcopy(self.picture)
        changed.coordinate_dependencies["N"]["point"] = "Beta0"
        cases.append(changed)
        changed = deepcopy(self.picture)
        changed.occlusion_relations[-1].face_names[-1] = "Alpha0"
        cases.append(changed)
        changed = deepcopy(self.picture)
        matrix = [list(row) for row in changed.projection_3d.matrix]
        matrix[0][0] += 0.125
        changed.projection_3d.matrix = tuple(tuple(row) for row in matrix)
        cases.append(changed)
        for case in cases:
            self.assertNotEqual(baseline, semantic_model_3d_hash(case))

    def test_revision_mismatch_blocks_motion_core_but_keeps_reviewable_candidates(self) -> None:
        rig = self._portable(
            analyze_geometry_rig_3d(self.picture, selection=self._selection()),
            expected_revision="source-sha256:" + "0" * 64,
        )
        self.assertFalse(rig["revisionMatch"])
        self.assertEqual(rig["status"], "blocked")
        self.assertIsNone(rig["motionSpecCore"])
        self.assertIsNone(rig["nativeManimSource"])
        self.assertIsNone(rig["nativeManimSourceV2"])
        self.assertTrue(rig["motionCandidates"])
        self.assertEqual(
            rig["diagnostics"][0]["code"],
            "PROVIDER_REVISION_MISMATCH",
        )
        Draft202012Validator(self.schema).validate(rig)

    def test_public_schema_accepts_a_ready_portable_result(self) -> None:
        rig = self._portable(
            analyze_geometry_rig_3d(self.picture, selection=self._selection())
        )
        Draft202012Validator(self.schema).validate(rig)
        self.assertTrue(rig["revisionMatch"])
        self.assertEqual(rig["status"], "ready")

    def test_public_schema_accepts_reviewable_discovery_without_motion_core(self) -> None:
        rig = self._portable(analyze_geometry_rig_3d(self.picture))
        Draft202012Validator(self.schema).validate(rig)
        self.assertEqual(rig["status"], "needs_selection")
        self.assertIsNone(rig["selectedMotionCandidate"])
        self.assertIsNone(rig["motionSpecCore"])
        self.assertIsNotNone(rig["nativeManimSourceV2"])
        self.assertTrue(rig["semanticGroups"])


if __name__ == "__main__":
    unittest.main()
