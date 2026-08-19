from __future__ import annotations

import unittest

import numpy as np

from polyhedron_visibility.dihedral_extraction import (
    BasePlaneRotation3D,
    DerivedDihedralModel,
    RigidTransform3D,
)
from polyhedron_visibility.dihedral_extraction.unified_compositing import (
    canonical_derived_dihedral_unified_compositing_json,
    compute_derived_dihedral_unified_compositing,
)

from tests.test_derived_dihedral_contract import cube_model
from tests.test_derived_dihedral_manim import isometric_projection
from examples.derived_dihedral_extraction.derived_dihedral_extraction_demo import (
    isometric_projection as demo_isometric_projection,
    rectangular_box,
    square_pyramid,
    tetrahedron,
)


class DerivedDihedralUnifiedCompositingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = DerivedDihedralModel.from_solid(
            "cube-unified-compositing",
            cube_model(),
            entity_id="copy",
            source_face_ids=("front", "top"),
        )

    def frame(self, shift: float):
        return compute_derived_dihedral_unified_compositing(
            self.model,
            transform=RigidTransform3D.translation_by((0.0, 0.0, shift)),
            projection_matrix=isometric_projection(),
        )

    def test_faces_and_strokes_share_one_exact_far_to_near_order(self) -> None:
        frame = self.frame(-0.5)
        ranks = {item_id: rank for rank, item_id in enumerate(frame.draw_order)}

        self.assertEqual(set(frame.draw_order), set(frame.item_ids))
        self.assertEqual(len(frame.draw_order), len(frame.item_ids))
        self.assertTrue(frame.face_batches)
        self.assertTrue(frame.stroke_fragments)
        for relation in frame.order_relations:
            self.assertLess(
                ranks[relation.far_item_id],
                ranks[relation.near_item_id],
            )

        kinds = tuple(
            "stroke" if item_id.startswith("stroke:") else "face"
            for item_id in frame.draw_order
        )
        transitions = sum(
            first != second for first, second in zip(kinds, kinds[1:])
        )
        self.assertGreaterEqual(transitions, 3)
        self.assertTrue(
            any(
                relation.far_item_id.startswith("stroke:")
                != relation.near_item_id.startswith("stroke:")
                for relation in frame.order_relations
            )
        )
        self.assertTrue(
            any(
                relation.far_item_id.startswith("stroke:")
                and relation.near_item_id.startswith("stroke:")
                for relation in frame.order_relations
            )
        )

    def test_refined_fragments_cover_each_active_semantic_stroke(self) -> None:
        frame = self.frame(-0.5)
        by_edge = {}
        for fragment in frame.stroke_fragments:
            by_edge.setdefault(fragment.source_edge_id, []).append(fragment)

        suppressed = set(frame.transparent.visibility.suppressed_source_stroke_ids)
        for stroke in self.model.overlay_model().strokes:
            if stroke.source_edge_id in suppressed:
                self.assertNotIn(stroke.source_edge_id, by_edge)
                continue
            fragments = sorted(
                by_edge[stroke.source_edge_id],
                key=lambda item: item.start_parameter,
            )
            self.assertAlmostEqual(fragments[0].start_parameter, 0.0)
            self.assertAlmostEqual(fragments[-1].end_parameter, 1.0)
            for first, second in zip(fragments, fragments[1:]):
                self.assertAlmostEqual(
                    first.end_parameter,
                    second.start_parameter,
                )
            for kind in ("visible", "hidden"):
                indices = sorted(
                    item.slot_index for item in fragments if item.slot_kind == kind
                )
                self.assertEqual(indices, list(range(len(indices))))

    def test_dense_motion_is_deterministic_and_never_emits_nan(self) -> None:
        for shift in np.linspace(0.0, -3.0, 25):
            frame = self.frame(float(shift))
            payload = canonical_derived_dihedral_unified_compositing_json(frame)
            repeated = canonical_derived_dihedral_unified_compositing_json(
                self.frame(float(shift))
            )
            self.assertEqual(payload, repeated)
            self.assertNotIn("NaN", payload)
            self.assertNotIn("Infinity", payload)

    def test_all_demo_separation_and_base_rotation_paths_remain_orderable(self) -> None:
        specs = (
            (
                rectangular_box(),
                ("front", "top"),
                "right",
                (2.6, -1.0, 1.0),
            ),
            (
                tetrahedron(),
                ("ABC", "ABD"),
                "ACD",
                (2.25, -0.9, 0.8),
            ),
            (
                square_pyramid(),
                ("side.AB", "side.BC"),
                "side.CD",
                (2.5, -0.8, 0.75),
            ),
        )
        for solid, source_faces, base_face, translation_value in specs:
            with self.subTest(solid=solid.visibility_group_id):
                model = DerivedDihedralModel.from_solid(
                    solid.visibility_group_id + "-unified-demo",
                    solid,
                    entity_id="copy",
                    source_face_ids=source_faces,
                )
                base_rotation = BasePlaneRotation3D.from_model(
                    solid,
                    base_face,
                )
                translation = np.asarray(translation_value, dtype=float)
                entry = solid.entry_positions

                samples = [
                    (progress, 0.0)
                    for progress in np.linspace(0.0, 1.0, 21)
                ] + [
                    (1.0, progress)
                    for progress in np.linspace(0.0, 1.0, 101)
                ] + [
                    (progress, 1.0)
                    for progress in np.linspace(1.0, 0.55, 11)
                ]
                maximum_slots = 0
                for separation, base_progress in samples:
                    source_shift = RigidTransform3D.translation_by(
                        -0.5 * translation * separation
                    )
                    global_transform = source_shift.compose(
                        base_rotation.transform(float(base_progress))
                    )
                    local_transform = RigidTransform3D.translation_by(
                        translation * separation
                    )
                    copy_transform = local_transform.compose(global_transform)
                    solid_positions = {
                        vertex_id: global_transform.apply(point)
                        for vertex_id, point in entry.items()
                    }
                    frame = compute_derived_dihedral_unified_compositing(
                        model,
                        transform=copy_transform,
                        projection_matrix=demo_isometric_projection(),
                        solid_vertex_positions=solid_positions,
                    )
                    counts = {}
                    for fragment in frame.stroke_fragments:
                        key = (fragment.source_edge_id, fragment.slot_kind)
                        counts[key] = counts.get(key, 0) + 1
                    maximum_slots = max(
                        maximum_slots,
                        max(counts.values(), default=0),
                    )
                self.assertLessEqual(maximum_slots, 12)


if __name__ == "__main__":
    unittest.main()
