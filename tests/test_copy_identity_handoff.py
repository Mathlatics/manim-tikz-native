from __future__ import annotations

import unittest

import numpy as np

from polyhedron_visibility.copy_handoff import (
    COPY_IDENTITY_HANDOFF_FRAME_SCHEMA,
    COPY_IDENTITY_HANDOFF_SCHEMA,
    CopyHandoffContractError,
    CopyIdentityHandoffMap,
    CopyIdentityHandoffPolicy,
    CopyPrimitivePair,
    CopyVertexPair,
    compute_copy_identity_handoff,
)
from tests.test_derived_dihedral_contract import cube_model


class CopyIdentityHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = cube_model()
        self.handoff = CopyIdentityHandoffMap.from_visibility_model(
            "cube-to-analysis-copy",
            self.model,
            source_entity_id="solid",
            copy_entity_id="analysis-copy",
            policy=CopyIdentityHandoffPolicy(activation_distance=0.2),
        )
        self.source_positions = {
            f"solid:{vertex_id}": np.asarray(vertex.entry_position, dtype=float)
            for vertex_id, vertex in self.model.vertex_map.items()
        }

    def copied_positions(self, offset=(0.0, 0.0, 0.0)):
        translation = np.asarray(offset, dtype=float)
        return {
            f"analysis-copy:{vertex_id}": np.asarray(
                vertex.entry_position, dtype=float
            )
            + translation
            for vertex_id, vertex in self.model.vertex_map.items()
        }

    def frame(self, offset=(0.0, 0.0, 0.0), **kwargs):
        return compute_copy_identity_handoff(
            self.handoff,
            source_positions=self.source_positions,
            copy_positions=self.copied_positions(offset),
            **kwargs,
        )

    def test_whole_solid_copy_builds_complete_semantic_lineage(self) -> None:
        self.assertEqual(self.handoff.schema, COPY_IDENTITY_HANDOFF_SCHEMA)
        self.assertEqual(len(self.handoff.vertex_pairs), len(self.model.vertices))
        self.assertEqual(len(self.handoff.face_pairs), len(self.model.faces))
        self.assertEqual(len(self.handoff.stroke_pairs), len(self.model.strokes))
        self.assertEqual(
            self.handoff.face_pairs[0].source_primitive_id,
            f"solid:{self.handoff.face_pairs[0].semantic_primitive_id}",
        )
        self.assertEqual(
            self.handoff.face_pairs[0].copy_primitive_id,
            f"analysis-copy:{self.handoff.face_pairs[0].semantic_primitive_id}",
        )

    def test_copy_owns_exact_identity_and_source_fades_in_smoothly(self) -> None:
        identity = self.frame()
        self.assertEqual(identity.schema, COPY_IDENTITY_HANDOFF_FRAME_SCHEMA)
        self.assertEqual(identity.maximum_separation, 0.0)
        self.assertEqual(identity.source_opacity_scale, 0.0)
        self.assertEqual(
            set(identity.source_face_opacity_scales.values()),
            {0.0},
        )
        self.assertEqual(
            set(identity.source_stroke_opacity_scales.values()),
            {0.0},
        )
        self.assertEqual(set(identity.copy_face_opacity_scales.values()), {1.0})
        self.assertEqual(set(identity.copy_stroke_opacity_scales.values()), {1.0})

        midpoint = self.frame((0.1, 0.0, 0.0))
        self.assertAlmostEqual(midpoint.maximum_separation, 0.1)
        self.assertAlmostEqual(midpoint.source_opacity_scale, 0.5)
        for value in midpoint.source_face_opacity_scales.values():
            self.assertAlmostEqual(value, 0.5)
        for value in midpoint.source_stroke_opacity_scales.values():
            self.assertAlmostEqual(value, 0.5)

        separated = self.frame((0.2, 0.0, 0.0))
        self.assertEqual(separated.source_opacity_scale, 1.0)
        self.assertEqual(set(separated.source_face_opacity_scales.values()), {1.0})
        self.assertEqual(set(separated.source_stroke_opacity_scales.values()), {1.0})

        returned = self.frame()
        self.assertEqual(returned.to_dict(), identity.to_dict())

    def test_projection_space_is_explicit_and_controls_the_handoff(self) -> None:
        copied = self.copied_positions((0.0, 0.0, 0.1))
        projected = compute_copy_identity_handoff(
            self.handoff,
            source_positions=self.source_positions,
            copy_positions=copied,
            final_point_provider=lambda point: (point[0], point[1], 0.0),
        )
        self.assertEqual(projected.maximum_separation, 0.0)
        self.assertEqual(projected.source_opacity_scale, 0.0)

        world = compute_copy_identity_handoff(
            self.handoff,
            source_positions=self.source_positions,
            copy_positions=copied,
        )
        self.assertAlmostEqual(world.maximum_separation, 0.1)
        self.assertAlmostEqual(world.source_opacity_scale, 0.5)

    def test_each_primitive_uses_its_own_corresponding_vertices(self) -> None:
        copied = self.copied_positions()
        copied["analysis-copy:E"] = copied["analysis-copy:E"] + (0.1, 0.0, 0.0)
        frame = compute_copy_identity_handoff(
            self.handoff,
            source_positions=self.source_positions,
            copy_positions=copied,
        )
        self.assertAlmostEqual(frame.source_face_opacity_scales["solid:front"], 0.5)
        self.assertAlmostEqual(frame.source_face_opacity_scales["solid:left"], 0.5)
        self.assertAlmostEqual(frame.source_face_opacity_scales["solid:bottom"], 0.5)
        self.assertEqual(frame.source_face_opacity_scales["solid:back"], 0.0)
        self.assertEqual(frame.source_face_opacity_scales["solid:right"], 0.0)
        self.assertEqual(frame.source_face_opacity_scales["solid:top"], 0.0)

    def test_selected_subset_and_invalid_lineage_fail_closed(self) -> None:
        subset = CopyIdentityHandoffMap.from_visibility_model(
            "front-copy",
            self.model,
            source_entity_id="solid",
            copy_entity_id="copy",
            face_ids=("front",),
            stroke_ids=("edge.E.F",),
        )
        self.assertEqual([item.semantic_primitive_id for item in subset.face_pairs], ["front"])
        self.assertEqual(
            [item.semantic_primitive_id for item in subset.stroke_pairs],
            ["edge.E.F"],
        )

        with self.assertRaisesRegex(
            CopyHandoffContractError,
            "MISSING_SOURCE_PRIMITIVE",
        ):
            CopyIdentityHandoffMap.from_visibility_model(
                "bad-copy",
                self.model,
                source_entity_id="solid",
                copy_entity_id="copy",
                face_ids=("missing",),
                stroke_ids=(),
            )

        with self.assertRaisesRegex(
            CopyHandoffContractError,
            "DUPLICATE_COPY_LINEAGE",
        ):
            CopyIdentityHandoffMap(
                "duplicate",
                (
                    CopyVertexPair("A", "solid:A", "copy:A"),
                    CopyVertexPair("A", "solid:B", "copy:B"),
                ),
                (
                    CopyPrimitivePair(
                        "face",
                        "solid:face",
                        "copy:face",
                        ("A", "B", "C"),
                    ),
                ),
                (),
            )

    def test_missing_or_invalid_runtime_positions_fail_closed(self) -> None:
        missing = self.copied_positions()
        del missing["analysis-copy:A"]
        with self.assertRaisesRegex(ValueError, "missing copy vertex position"):
            compute_copy_identity_handoff(
                self.handoff,
                source_positions=self.source_positions,
                copy_positions=missing,
            )

        invalid = self.copied_positions()
        invalid["analysis-copy:A"] = np.asarray((np.nan, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "finite three-component point"):
            compute_copy_identity_handoff(
                self.handoff,
                source_positions=self.source_positions,
                copy_positions=invalid,
            )

        with self.assertRaisesRegex(TypeError, "final_point_provider"):
            compute_copy_identity_handoff(
                self.handoff,
                source_positions=self.source_positions,
                copy_positions=self.copied_positions(),
                final_point_provider=object(),
            )

        for value in (0.0, -0.1, float("nan"), True):
            with self.subTest(value=value), self.assertRaisesRegex(
                CopyHandoffContractError,
                "INVALID_ACTIVATION_DISTANCE",
            ):
                CopyIdentityHandoffPolicy(activation_distance=value)


if __name__ == "__main__":
    unittest.main()
