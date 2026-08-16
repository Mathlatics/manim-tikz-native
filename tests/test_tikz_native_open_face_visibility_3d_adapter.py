from __future__ import annotations

import copy
from pathlib import Path
import unittest

import numpy as np

from tikz_native.compiler import compile_document
from tikz_native.open_face_visibility_3d_adapter import (
    OPEN_FACE_ADAPTER_RESULT_SCHEMA,
    TikzNativeOpenFaceVisibility3DAdapterError,
    adapt_picture_open_face_visibility_3d,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "dihedral_fold_3d_demo" / "dihedral_fold.tex"
PLAIN_OPEN_FACES = r"""
\begin{tikzpicture}[3d view={40.4}{23.8}]
  \coordinate (A) at (0,-1.8,0);
  \coordinate (B) at (0,1.8,0);
  \coordinate (Alpha0) at (3.4,-1.8,0);
  \coordinate (Alpha1) at (3.4,1.8,0);
  \coordinate (Beta0) at (1.8017254984,-1.8,2.8833635269);
  \coordinate (Beta1) at (1.8017254984,1.8,2.8833635269);
  \coordinate (S) at (-1,0,1.5);
  \coordinate (E) at (4,0,1.5);
  \fill[fill opacity=.3] (A)--(B)--(Alpha1)--(Alpha0)--cycle;
  \fill[fill opacity=.3] (A)--(B)--(Beta1)--(Beta0)--cycle;
  \DeclareSpaceHinge{fold-angle}{A/B}{A/B/Alpha1/Alpha0}{A/B/Beta1/Beta0}
  \draw[purple] (S)--(E);
\end{tikzpicture}
"""


def _dihedral_source_with_beta(*, x: float, z: float) -> str:
    return (
        SOURCE.read_text(encoding="utf-8")
        .replace(
            "(1.8017254984,-1.8,2.8833635269)",
            f"({x},-1.8,{z})",
        )
        .replace(
            "(1.8017254984,1.8,2.8833635269)",
            f"({x},1.8,{z})",
        )
    )


class TikzNativeOpenFaceVisibility3DAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.picture = compile_document(SOURCE).pictures[0]

    def test_real_dihedral_proves_two_faces_one_hinge_and_nine_logical_strokes(self) -> None:
        result = adapt_picture_open_face_visibility_3d(self.picture)

        self.assertEqual(result.schema, OPEN_FACE_ADAPTER_RESULT_SCHEMA)
        self.assertEqual(len(result.model.faces), 2)
        self.assertEqual(len(result.model.seams), 1)
        self.assertEqual(len(result.model.strokes), 9)
        self.assertEqual(result.model.seams[0].seam_id, "fold-angle")
        self.assertEqual(result.model.seams[0].policy, "articulated_hinge")
        self.assertEqual(set(result.model.seams[0].vertex_ids), {"A", "B"})

        beta_face = next(
            face.face_id
            for face in result.face_bindings
            if any("Beta1" in cycle for cycle in face.authored_cycles)
        )
        mn = next(
            stroke for stroke in result.model.strokes if set(stroke.vertex_ids) == {"M", "N"}
        )
        se = next(
            stroke for stroke in result.model.strokes if set(stroke.vertex_ids) == {"S", "E"}
        )
        self.assertEqual(se.vertex_ids, ("S", "E"))
        self.assertEqual(mn.incident_face_ids, ())
        self.assertEqual(mn.excluded_occluder_face_ids, (beta_face,))
        self.assertEqual(se.excluded_occluder_face_ids, ())

        se_proof = next(
            proof
            for proof in result.relation_proofs
            if proof.authored_vertex_ids == ("S", "E")
        )
        self.assertEqual(len(se_proof.fragments), 3)
        self.assertEqual(
            [item.visibility for item in se_proof.fragments],
            ["visible", "hidden", "visible"],
        )
        self.assertAlmostEqual(se_proof.fragments[0].start_parameter, 0.0)
        self.assertAlmostEqual(se_proof.fragments[-1].end_parameter, 1.0)
        for first, second in zip(se_proof.fragments, se_proof.fragments[1:]):
            self.assertAlmostEqual(first.end_parameter, second.start_parameter)

        ab_proofs = [
            proof
            for proof in result.relation_proofs
            if set(proof.canonical_vertex_ids) == {"A", "B"}
        ]
        self.assertEqual(len(ab_proofs), 2)
        self.assertEqual(len({proof.source_edge_id for proof in ab_proofs}), 1)

        spans = result.entry_trace.edge_map[se.source_edge_id].spans
        self.assertEqual([span.kind for span in spans], ["visible", "hidden", "visible"])
        self.assertEqual(spans[1].occluder_face_ids, (beta_face,))

    def test_plain_complete_line_is_occluded_by_global_faces_without_legacy_answers(self) -> None:
        picture = compile_document(source_text=PLAIN_OPEN_FACES).pictures[0]
        self.assertFalse(picture.occlusion_relations)

        result = adapt_picture_open_face_visibility_3d(picture)
        self.assertFalse(result.relation_proofs)
        self.assertEqual(len(result.model.faces), 2)
        self.assertEqual(len(result.model.strokes), 1)
        stroke = result.model.strokes[0]
        self.assertEqual(stroke.vertex_ids, ("S", "E"))
        self.assertEqual(
            [span.kind for span in result.entry_trace.edge_map[stroke.source_edge_id].spans],
            ["visible", "hidden", "visible"],
        )

    def test_exact_and_near_flat_hinges_keep_two_authored_faces_at_original_positions(self) -> None:
        cases = (
            ("exact-zero", 3.4, 0.0, "coplanar_same_normal", 0.0),
            ("near-zero", 3.4, 1.0e-11, "coplanar_same_normal", 0.0),
            ("exact-pi", -3.4, 0.0, "coplanar_opposite_normal", np.pi),
            ("near-pi", -3.4, 1.0e-11, "coplanar_opposite_normal", np.pi),
        )
        for label, x_value, z_value, state, angle in cases:
            with self.subTest(label=label):
                picture = compile_document(
                    source_text=_dihedral_source_with_beta(x=x_value, z=z_value)
                ).pictures[0]
                result = adapt_picture_open_face_visibility_3d(picture)

                self.assertEqual(len(result.model.faces), 2)
                self.assertEqual(len(result.model.seams), 1)
                self.assertNotEqual(
                    result.coordinate_vertex_map["Alpha0"],
                    result.coordinate_vertex_map["Beta0"],
                )
                self.assertEqual(
                    result.model.entry_positions[result.coordinate_vertex_map["Beta0"]],
                    picture.coordinates["Beta0"],
                )
                seam = result.entry_trace.seam_states[0]
                self.assertEqual(seam.state, state)
                self.assertAlmostEqual(seam.dihedral_radians, angle)
                # The legacy evidence is intentionally topology-only: its
                # private probe coordinates and trace cannot leak to callers.
                legacy_payload = result.legacy_analysis.to_dict()
                self.assertNotIn("model", legacy_payload)
                self.assertNotIn("entryTrace", legacy_payload)

    def test_tikz_adapter_does_not_expose_a_second_tolerance_contract(self) -> None:
        with self.assertRaisesRegex(TypeError, "tolerance_policy"):
            adapt_picture_open_face_visibility_3d(
                self.picture,
                tolerance_policy=object(),  # type: ignore[call-arg]
            )

    def test_relation_parameter_gap_fails_before_open_model_creation(self) -> None:
        picture = copy.deepcopy(self.picture)
        relation = next(item for item in picture.occlusion_relations if item.start_name == "S")
        first = next(item for item in picture.objects if item.id == relation.object_ids[0])
        old_end = float(first.geometry["source_parameter_range"][1])
        new_end = old_end - 0.01
        first.geometry["source_parameter_range"] = (0.0, new_end)
        start = np.asarray(picture.coordinates[relation.start_name], dtype=float)
        end = np.asarray(picture.coordinates[relation.end_name], dtype=float)
        first.geometry["end"] = tuple(start + new_end * (end - start))

        with self.assertRaises(TikzNativeOpenFaceVisibility3DAdapterError) as caught:
            adapt_picture_open_face_visibility_3d(picture)
        self.assertEqual(caught.exception.code, "INVALID_FRAGMENT_PARTITION")
        self.assertIn("gap", str(caught.exception))

    def test_off_line_fragment_fails_closed(self) -> None:
        picture = copy.deepcopy(self.picture)
        item = next(value for value in picture.objects if value.id == "occluded_visible.S.E.0")
        start = np.asarray(item.geometry["start"], dtype=float)
        item.geometry["start"] = tuple(start + np.asarray((0.0, 0.02, 0.0)))

        with self.assertRaises(TikzNativeOpenFaceVisibility3DAdapterError) as caught:
            adapt_picture_open_face_visibility_3d(picture)
        self.assertEqual(caught.exception.code, "NONCOLLINEAR_RELATION_FRAGMENT")

    def test_fragment_visibility_and_style_are_compiler_proven(self) -> None:
        for mutation, expected_code in (
            ("visibility", "INVALID_FRAGMENT_VISIBILITY"),
            ("style", "FRAGMENT_STYLE_MISMATCH"),
        ):
            with self.subTest(mutation=mutation):
                picture = copy.deepcopy(self.picture)
                item = next(
                    value for value in picture.objects if value.id == "occluded_visible.S.E.0"
                )
                if mutation == "visibility":
                    item.geometry["visibility"] = "invented"
                else:
                    # Compiler output intentionally shares the immutable style
                    # value with its relation.  Detach only the object copy so
                    # this test models corrupted fragment evidence.
                    item.style = copy.deepcopy(item.style)
                    item.style.line_width_pt += 0.25
                with self.assertRaises(TikzNativeOpenFaceVisibility3DAdapterError) as caught:
                    adapt_picture_open_face_visibility_3d(picture)
                self.assertEqual(caught.exception.code, expected_code)

    def test_multiple_plain_named_lines_are_not_promoted_to_fragment_evidence(self) -> None:
        picture = copy.deepcopy(self.picture)
        source = next(item for item in picture.objects if item.id == "line.M.N")
        duplicate = copy.deepcopy(source)
        duplicate.id = "line.M.N.duplicate"
        picture.objects.append(duplicate)

        with self.assertRaises(TikzNativeOpenFaceVisibility3DAdapterError) as caught:
            adapt_picture_open_face_visibility_3d(picture)
        self.assertEqual(caught.exception.code, "AMBIGUOUS_STROKE_SOURCES")

    def test_canonical_result_survives_harmless_compiler_list_reordering(self) -> None:
        reordered = copy.deepcopy(self.picture)
        reordered.objects.reverse()
        reordered.occlusion_relations.reverse()
        reordered.hinge_relations.reverse()
        reordered.coordinates = dict(reversed(tuple(reordered.coordinates.items())))

        first = adapt_picture_open_face_visibility_3d(self.picture)
        second = adapt_picture_open_face_visibility_3d(reordered)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.result_sha256, second.result_sha256)


if __name__ == "__main__":
    unittest.main()
