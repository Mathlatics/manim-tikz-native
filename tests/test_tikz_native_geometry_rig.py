from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

from tikz_native.compiler import compile_document
from tikz_native.geometry_rig import (
    GEOMETRY_RIG_SCHEMA,
    GeometryRigError,
    analyze_geometry_rig,
    attach_geometry_rig_identity,
    motion_spec_payload,
    semantic_model_hash,
)
from tikz_native.motion_runtime import MotionSpec
from tikz_native.provider import provider_info, sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "analytic_geometry_ellipse_demo"
SOURCE = DEMO / "ellipse_problem.tex"
SCHEMA = ROOT / "tikz_native" / "schemas" / "geometry-rig-v1.schema.json"


class TikzNativeGeometryRigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = compile_document(SOURCE)
        cls.picture = cls.document.pictures[0]
        cls.schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)

    def _portable(self, rig: dict, *, expected_revision: str | None = None) -> dict:
        revision = provider_info()["revision"]
        return attach_geometry_rig_identity(
            rig,
            source_sha256=sha256_file(SOURCE),
            provider_revision=revision,
            expected_asset_provider_revision=expected_revision or revision,
        )

    def test_demo_active_line_infers_exact_motion_bindings(self) -> None:
        rig = analyze_geometry_rig(self.picture, "line.Lstart.Lend")
        self.assertEqual(rig["schema"], GEOMETRY_RIG_SCHEMA)
        self.assertEqual(rig["status"], "ready")
        driver = rig["selectedDriver"]
        self.assertEqual(driver["candidateId"], "rotate_named_line:Lpath:0")
        self.assertEqual(driver["activePath"], "Lpath")
        self.assertEqual(driver["pivot"], "F")
        self.assertEqual(driver["intersectionIndex"], 0)
        self.assertAlmostEqual(driver["initial"], 0.6435011087932844)
        self.assertLess(driver["range"][0], driver["initial"])
        self.assertGreater(driver["range"][1], driver["initial"])

        included = {
            item["objectId"] for item in rig["bindings"] if item["status"] == "included"
        }
        self.assertEqual(
            included,
            {
                "line.Lstart.Lend",
                "label.Lend.l",
                "fill.P.Q.R",
                "fill.P.F.O",
                "line.P.Q",
                "line.Q.R",
                "line.R.P",
                "line.P.F",
                "line.O.P",
                "line.Q.O",
                "angle.R.Q.P",
                "label_angle.R.Q.P.varphi",
                "dot.P",
                "dot.Q",
                "dot.R",
                "label.P.P",
                "label.Q.Q",
                "label.R.R",
            },
        )
        self.assertEqual(len(rig["motionSpecCore"]["bindings"]), 18)
        self.assertNotIn("schema", rig["motionSpecCore"])
        self.assertNotIn("timeline", rig["motionSpecCore"])
        self.assertEqual(
            [
                item["objectId"]
                for item in rig["activeObjectCandidates"]
                if item["status"] == "available"
            ],
            ["line.Lstart.Lend"],
        )

    def test_coordinate_dependencies_distinguish_fixed_intersection_and_derived(self) -> None:
        rig = analyze_geometry_rig(self.picture, "line.Lstart.Lend")
        records = {item["coordinateId"]: item for item in rig["coordinates"]}
        self.assertEqual(records["F"]["classification"], "fixed")
        self.assertFalse(records["F"]["affectedByDriver"])
        self.assertEqual(records["Lstart"]["classification"], "driver_endpoint")
        self.assertTrue(records["Lstart"]["affectedByDriver"])
        self.assertEqual(records["P"]["classification"], "intersection")
        self.assertEqual(records["Q"]["classification"], "intersection")
        self.assertEqual(records["R"]["classification"], "derived")
        self.assertEqual(records["R"]["dependsOn"], ["O", "P"])
        self.assertTrue(records["R"]["affectedByDriver"])
        self.assertEqual(rig["intersections"][0]["status"], "driven")
        self.assertIn("F", rig["fixedCoordinateIds"])
        self.assertNotIn("P", rig["fixedCoordinateIds"])

    def test_derived_midpoint_on_active_line_is_not_a_pivot_candidate(self) -> None:
        picture = deepcopy(self.picture)
        focus = picture.coordinates["F"]
        guide = picture.coordinates["Pguide"]
        picture.coordinates["derivedMidpoint"] = (
            (focus[0] + guide[0]) / 2,
            (focus[1] + guide[1]) / 2,
        )
        picture.coordinate_dependencies["derivedMidpoint"] = {
            "operation": "interpolation",
            "start": "F",
            "end": "Pguide",
            "parameter": 0.5,
            "parameter_expression": "0.5",
        }

        rig = analyze_geometry_rig(picture, "line.Lstart.Lend")
        pivot_ids = {
            pivot["coordinateId"]
            for candidate in rig["driverCandidates"]
            for pivot in candidate["pivotCandidates"]
        }
        self.assertNotIn("derivedMidpoint", pivot_ids)
        self.assertEqual(rig["selectedDriver"]["pivot"], "F")
        derived = next(
            item
            for item in rig["coordinates"]
            if item["coordinateId"] == "derivedMidpoint"
        )
        self.assertEqual(derived["classification"], "derived")
        self.assertFalse(derived["affectedByDriver"])

    def test_ready_core_completes_and_validates_as_existing_motion_spec(self) -> None:
        rig = analyze_geometry_rig(
            self.picture,
            "line.Lstart.Lend",
            selection={
                "candidate_id": "rotate_named_line:Lpath:0",
                "pivot": "F",
                "range": [0.4, 0.9],
            },
        )
        payload = motion_spec_payload(rig)
        self.assertEqual(payload["schema"], "tikz-native-motion/v1")
        self.assertEqual(payload["timeline"], [])
        spec = MotionSpec.from_dict(payload)
        spec.validate_picture(self.picture)
        self.assertEqual(spec.driver.active_path, "Lpath")
        self.assertEqual(len(spec.bindings), 18)

    def test_excluded_dependency_remains_visible_but_is_not_compiled(self) -> None:
        rig = analyze_geometry_rig(
            self.picture,
            "line.Lstart.Lend",
            selection={"exclude_object_ids": ["label.R.R"]},
        )
        self.assertEqual(rig["status"], "ready")
        self.assertEqual(rig["excludedObjectIds"], ["label.R.R"])
        dependency = next(
            item
            for item in rig["rigDraft"]["dependencies"]
            if item["objectId"] == "label.R.R"
        )
        self.assertEqual(dependency["status"], "excluded")
        self.assertFalse(dependency["enabled"])
        compiled_ids = {
            item["object_id"] for item in rig["motionSpecCore"]["bindings"]
        }
        self.assertNotIn("label.R.R", compiled_ids)
        self.assertIn(
            "DEPENDENT_OBJECTS_EXCLUDED",
            {item["code"] for item in rig["diagnostics"]},
        )

    def test_invalid_selection_and_active_object_fail_closed(self) -> None:
        with self.assertRaisesRegex(GeometryRigError, "active object cannot be excluded"):
            analyze_geometry_rig(
                self.picture,
                "line.Lstart.Lend",
                selection={"exclude_object_ids": ["line.Lstart.Lend"]},
            )
        with self.assertRaisesRegex(GeometryRigError, "must retain the active object"):
            analyze_geometry_rig(
                self.picture,
                "line.Lstart.Lend",
                selection={"include_object_ids": ["dot.P"]},
            )
        with self.assertRaisesRegex(GeometryRigError, "unsupported fields"):
            analyze_geometry_rig(
                self.picture,
                "line.Lstart.Lend",
                selection={"python": "arbitrary"},
            )

        missing = analyze_geometry_rig(self.picture, "missing.object")
        self.assertEqual(missing["status"], "blocked")
        self.assertIsNone(missing["motionSpecCore"])
        self.assertEqual(missing["diagnostics"][0]["code"], "ACTIVE_OBJECT_NOT_FOUND")

        dot = analyze_geometry_rig(self.picture, "dot.P")
        self.assertEqual(dot["status"], "blocked")
        self.assertEqual(
            dot["diagnostics"][0]["code"], "ACTIVE_OBJECT_KIND_UNSUPPORTED"
        )

    def test_unselected_intersection_using_active_path_blocks_motion_core(self) -> None:
        picture = deepcopy(self.picture)
        picture.intersections.append(deepcopy(picture.intersections[0]))
        rig = analyze_geometry_rig(picture, "line.Lstart.Lend")
        self.assertEqual(rig["status"], "blocked")
        self.assertIsNone(rig["motionSpecCore"])
        self.assertIn(
            "UNSELECTED_ACTIVE_INTERSECTION",
            {item["code"] for item in rig["diagnostics"]},
        )
        self.assertEqual(rig["intersections"][1]["status"], "unsupported")

    def test_semantic_model_hash_is_portable_and_sensitive_to_relations(self) -> None:
        from_path = semantic_model_hash(self.picture)
        from_text = semantic_model_hash(
            compile_document(source_text=SOURCE.read_text(encoding="utf-8")).pictures[0]
        )
        self.assertEqual(from_path, from_text)
        changed = deepcopy(self.picture)
        changed.coordinate_dependencies["R"]["parameter"] = -0.5
        self.assertNotEqual(from_path, semantic_model_hash(changed))

    def test_provider_revision_mismatch_is_explicit_but_keeps_analysis_draft(self) -> None:
        rig = self._portable(
            analyze_geometry_rig(self.picture, "line.Lstart.Lend"),
            expected_revision="source-sha256:" + "0" * 64,
        )
        self.assertFalse(rig["revisionMatch"])
        self.assertEqual(rig["status"], "ready")
        self.assertIsNotNone(rig["motionSpecCore"])
        self.assertEqual(
            rig["diagnostics"][0]["code"], "PROVIDER_REVISION_MISMATCH"
        )
        Draft202012Validator(self.schema).validate(rig)

    def test_public_rig_schema_validates_full_ready_result(self) -> None:
        rig = self._portable(analyze_geometry_rig(self.picture, "line.Lstart.Lend"))
        Draft202012Validator(self.schema).validate(rig)
        self.assertEqual(rig["sourceSha256"], sha256_file(SOURCE))
        self.assertTrue(rig["revisionMatch"])


if __name__ == "__main__":
    unittest.main()
