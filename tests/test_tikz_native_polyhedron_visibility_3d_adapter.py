from __future__ import annotations

import copy
import unittest
from pathlib import Path

from tikz_native import compile_document
from tikz_native.polyhedron_visibility_3d_adapter import (
    ADAPTER_RESULT_SCHEMA,
    TikzNativeVisibility3DAdapterError,
    adapt_picture_visibility_3d,
)


PROVIDER_ROOT = Path(__file__).resolve().parents[1]
DIHEDRAL_SOURCE = (
    PROVIDER_ROOT / "examples" / "dihedral_fold_3d_demo" / "dihedral_fold.tex"
)


TRIANGULATED_SQUARE = r"""
\begin{tikzpicture}[3d view={40.4}{23.8}]
  \coordinate (A) at (-1,-1,1);
  \coordinate (B) at (1,-1,1);
  \coordinate (C) at (1,1,1);
  \coordinate (D) at (-1,1,1);
  \coordinate (S) at (-2,0,0);
  \coordinate (E) at (2,0,0);
  \fill[fill opacity=0.12] (A)--(B)--(C)--cycle;
  \fill[fill opacity=0.12] (A)--(C)--(D)--cycle;
  \draw (S)--(E);
\end{tikzpicture}
"""


ARROW_SOURCE = r"""
\begin{tikzpicture}[3d view={40.4}{23.8}]
  \coordinate (A) at (-1,-1,1);
  \coordinate (B) at (1,-1,1);
  \coordinate (C) at (1,1,1);
  \coordinate (D) at (-1,1,1);
  \coordinate (S) at (-2,0,0);
  \coordinate (E) at (2,0,0);
  \fill[fill opacity=0.12] (A)--(B)--(C)--(D)--cycle;
  \draw (S)--(E);
\end{tikzpicture}
"""


class TikzNativePolyhedronVisibility3DAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.picture = compile_document(DIHEDRAL_SOURCE).pictures[0]

    def test_closed_mode_rejects_real_dihedral_as_an_open_face_system(self) -> None:
        with self.assertRaises(TikzNativeVisibility3DAdapterError) as caught:
            adapt_picture_visibility_3d(
                self.picture,
                validation_mode="closed_convex_polyhedron",
            )

        self.assertEqual(caught.exception.code, "OPEN_FACE_SYSTEM")
        self.assertIn("closed", str(caught.exception).lower())

    def test_independent_mode_recovers_faces_full_relations_and_plain_line(self) -> None:
        result = adapt_picture_visibility_3d(
            self.picture,
            validation_mode="independent_convex_faces",
        )

        self.assertEqual(result.schema, ADAPTER_RESULT_SCHEMA)
        self.assertEqual(len(result.model.faces), 2)
        self.assertEqual(len(result.model.strokes), 9)
        self.assertEqual(len(result.face_bindings), 2)
        self.assertEqual(len(result.stroke_bindings), 9)

        stroke_by_vertices = {
            frozenset(stroke.vertex_ids): stroke
            for stroke in result.model.strokes
        }
        self.assertEqual(stroke_by_vertices[frozenset(("S", "E"))].incident_face_ids, ())
        self.assertEqual(stroke_by_vertices[frozenset(("M", "N"))].incident_face_ids, ())

        binding = next(
            item
            for item in result.stroke_bindings
            if frozenset(item.vertex_ids) == frozenset(("S", "E"))
        )
        self.assertEqual(binding.source_kind, "legacy_occlusion_relation")
        self.assertEqual(len(binding.object_ids), 3)
        self.assertEqual(binding.vertex_ids, tuple(sorted(("S", "E"))))
        self.assertTrue(set(binding.object_ids).issubset(result.suppressed_object_ids))

        plain = next(
            item
            for item in result.stroke_bindings
            if frozenset(item.vertex_ids) == frozenset(("M", "N"))
        )
        self.assertEqual(plain.source_kind, "named_line")
        self.assertEqual(plain.object_ids, ("line.M.N",))
        self.assertIn("line.M.N", result.suppressed_object_ids)

        self.assertEqual(result.entry_projection, self.picture.projection_3d.matrix)
        self.assertEqual(result.entry_trace.visibility_group_id, result.model.visibility_group_id)
        self.assertEqual(len(result.model_sha256), 64)
        self.assertEqual(len(result.entry_trace_sha256), 64)
        self.assertEqual(len(result.result_sha256), 64)

    def test_faces_occlude_independently_of_fill_alpha_and_accept_explicit_override(self) -> None:
        picture = compile_document(source_text=TRIANGULATED_SQUARE).pictures[0]
        result = adapt_picture_visibility_3d(
            picture,
            validation_mode="independent_convex_faces",
        )
        self.assertEqual(len(result.model.faces), 1)
        self.assertTrue(result.model.faces[0].occludes_strokes)
        self.assertEqual(len(result.face_bindings[0].object_ids), 2)
        self.assertEqual(len(result.face_bindings[0].authored_cycles), 2)
        self.assertEqual(len(result.model.faces[0].vertex_ids), 4)

        first_polygon_id = result.face_bindings[0].object_ids[0]
        overridden = adapt_picture_visibility_3d(
            picture,
            validation_mode="independent_convex_faces",
            overrides={
                "faceOccludesStrokes": {first_polygon_id: False},
            },
        )
        self.assertFalse(overridden.model.faces[0].occludes_strokes)

    def test_relation_members_are_not_reintroduced_as_fragment_strokes(self) -> None:
        result = adapt_picture_visibility_3d(
            self.picture,
            validation_mode="independent_convex_faces",
        )
        stroke_object_ids = {
            object_id
            for binding in result.stroke_bindings
            for object_id in binding.object_ids
        }
        self.assertIn("occluded_hidden.S.E.1", stroke_object_ids)
        self.assertNotIn(
            "occluded_hidden.S.E.1",
            {stroke.source_edge_id for stroke in result.model.strokes},
        )

    def test_reversed_duplicate_polygon_cycles_are_one_bound_face(self) -> None:
        picture = compile_document(source_text=ARROW_SOURCE).pictures[0]
        polygon = next(item for item in picture.objects if item.kind == "polygon")
        duplicate = copy.deepcopy(polygon)
        duplicate.id = "fill.reversed-duplicate"
        duplicate.geometry["point_names"] = list(
            reversed(duplicate.geometry["point_names"])
        )
        duplicate.geometry["points"] = list(reversed(duplicate.geometry["points"]))
        picture.objects.append(duplicate)

        result = adapt_picture_visibility_3d(
            picture,
            validation_mode="independent_convex_faces",
        )

        self.assertEqual(len(result.model.faces), 1)
        self.assertEqual(
            set(result.face_bindings[0].object_ids),
            {polygon.id, duplicate.id},
        )

    def test_coincident_coordinate_aliases_share_one_core_vertex(self) -> None:
        source = ARROW_SOURCE.replace(
            r"\coordinate (S) at (-2,0,0);",
            "\n".join(
                (
                    r"\coordinate (S) at (-2,0,0);",
                    r"\coordinate (S_alias) at (-2,0,0);",
                )
            ),
        ).replace(r"\draw (S)--(E);", r"\draw (S_alias)--(E);")
        picture = compile_document(source_text=source).pictures[0]

        result = adapt_picture_visibility_3d(
            picture,
            validation_mode="independent_convex_faces",
        )

        self.assertEqual(
            result.coordinate_vertex_map["S"],
            result.coordinate_vertex_map["S_alias"],
        )
        self.assertEqual(
            len(result.model.vertices),
            len({item.entry_position for item in result.model.vertices}),
        )
        self.assertIn(
            "WELDED_COORDINATE_ALIASES",
            {item.code for item in result.diagnostics},
        )

    def test_arrow_line_is_explicitly_unmanaged_instead_of_silently_rebuilt(self) -> None:
        picture = compile_document(source_text=ARROW_SOURCE).pictures[0]
        arrow = next(item for item in picture.objects if item.kind == "line")
        # The restricted compiler currently rejects TikZ arrow syntax before
        # producing an ObjectSpec.  Exercise the adapter's independent safety
        # gate with the equivalent compiled semantic value.
        arrow.style.arrow_tip = "Stealth"
        result = adapt_picture_visibility_3d(
            picture,
            validation_mode="independent_convex_faces",
        )
        self.assertIn(arrow.id, result.unmanaged_object_ids)
        self.assertFalse(result.model.strokes)
        self.assertIn(
            "UNMANAGED_ARROW_STROKE",
            {item.code for item in result.diagnostics},
        )

    def test_result_is_canonical_under_picture_list_reordering(self) -> None:
        reordered = copy.deepcopy(self.picture)
        reordered.objects.reverse()
        reordered.occlusion_relations.reverse()
        reordered.hinge_relations.reverse()
        reordered.coordinates = dict(reversed(tuple(reordered.coordinates.items())))

        first = adapt_picture_visibility_3d(
            self.picture,
            validation_mode="independent_convex_faces",
        )
        second = adapt_picture_visibility_3d(
            reordered,
            validation_mode="independent_convex_faces",
        )

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.result_sha256, second.result_sha256)

    def test_unknown_override_target_fails_closed(self) -> None:
        with self.assertRaises(TikzNativeVisibility3DAdapterError) as caught:
            adapt_picture_visibility_3d(
                self.picture,
                validation_mode="independent_convex_faces",
                overrides={"faceOccludesStrokes": {"missing-face": False}},
            )
        self.assertEqual(caught.exception.code, "UNKNOWN_OVERRIDE_TARGET")


if __name__ == "__main__":
    unittest.main()
