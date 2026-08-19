from __future__ import annotations

import unittest

from polyhedron_visibility.dihedral_extraction import (
    DerivedDihedralModel,
    RigidTransform3D,
    canonical_derived_dihedral_compositing_json,
    compute_derived_dihedral_transparent_compositing,
)

from tests.test_derived_dihedral_contract import cube_model
from tests.test_derived_dihedral_manim import isometric_projection


class DerivedDihedralTransparentCompositingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = DerivedDihedralModel.from_solid(
            "cube-transparent-extraction",
            cube_model(),
            entity_id="copy",
            source_face_ids=("front", "top"),
        )

    def test_identity_uses_derived_face_once_without_alpha_double_counting(self) -> None:
        frame = compute_derived_dihedral_transparent_compositing(
            self.model,
            transform=RigidTransform3D.identity(),
            projection_matrix=isometric_projection(),
        )

        surfaces = {item.surface_id for item in frame.fragments}
        self.assertNotIn("solid:front", surfaces)
        self.assertNotIn("solid:top", surfaces)
        self.assertIn("copy:front", surfaces)
        self.assertIn("copy:top", surfaces)
        self.assertEqual(set(frame.draw_order), set(frame.fragment_map))

    def test_intersecting_motion_splits_both_entities_and_orders_every_fragment(self) -> None:
        frame = compute_derived_dihedral_transparent_compositing(
            self.model,
            transform=RigidTransform3D.translation_by((0.4, 0.2, -0.5)),
            projection_matrix=isometric_projection(),
        )

        self.assertEqual(frame.visibility.coincident_source_face_ids, ())
        self.assertGreater(len(frame.fragments), 16)
        self.assertTrue(
            any(item.role == "solid_face" for item in frame.fragments)
        )
        self.assertTrue(
            any(item.role == "section_inside" for item in frame.fragments)
        )
        self.assertTrue(frame.order_relations)
        self.assertEqual(len(frame.draw_order), len(frame.fragments))
        payload = canonical_derived_dihedral_compositing_json(frame)
        self.assertEqual(payload, canonical_derived_dihedral_compositing_json(frame))
        self.assertNotIn("NaN", payload)

    def test_disjoint_finite_faces_are_not_split_by_supporting_planes(self) -> None:
        frame = compute_derived_dihedral_transparent_compositing(
            self.model,
            transform=RigidTransform3D.translation_by((5.0, 0.0, 0.0)),
            projection_matrix=isometric_projection(),
        )

        # Eight quads remain intact and therefore need only their ordinary two
        # triangles each.  The previous infinite-plane gate produced extra
        # fragments even though the finite source and copied faces were apart.
        self.assertEqual(len(frame.fragments), 16)
        self.assertTrue(all(len(batch) >= 1 for batch in frame.draw_batches))
        self.assertEqual(
            tuple(fragment_id for batch in frame.draw_batches for fragment_id in batch),
            frame.draw_order,
        )
        for batch in frame.draw_batches:
            self.assertEqual(
                len(
                    {
                        frame.fragment_map[fragment_id].source_face_id
                        for fragment_id in batch
                    }
                ),
                1,
            )

        copied_batches = [
            batch
            for batch in frame.draw_batches
            if frame.fragment_map[batch[0]].source_face_id in {"copy:front", "copy:top"}
        ]
        self.assertEqual([len(batch) for batch in copied_batches], [2, 2])


if __name__ == "__main__":
    unittest.main()
