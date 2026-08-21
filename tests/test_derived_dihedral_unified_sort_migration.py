from __future__ import annotations

import heapq
import random
import unittest
from unittest.mock import patch

import numpy as np

from polyhedron_visibility.compositor import stable_topological_sort
from polyhedron_visibility.dihedral_extraction import (
    DerivedDihedralModel,
    RigidTransform3D,
)
from polyhedron_visibility.dihedral_extraction.unified_compositing import (
    DerivedDihedralUnifiedCompositingError,
    UnifiedPaintRelation,
    _draw_order,
    compute_derived_dihedral_unified_compositing,
)

from tests.test_derived_dihedral_contract import cube_model


def _relation(far: str, near: str) -> UnifiedPaintRelation:
    return UnifiedPaintRelation(
        far,
        near,
        "test",
        -1.0,
        1.0,
        1.0,
    )


def _legacy_draw_order(
    item_ids: tuple[str, ...],
    relations: tuple[UnifiedPaintRelation, ...],
) -> tuple[str, ...]:
    identities = set(item_ids)
    outgoing = {item_id: set() for item_id in identities}
    indegree = {item_id: 0 for item_id in identities}
    for relation in relations:
        if (
            relation.far_item_id not in identities
            or relation.near_item_id not in identities
        ):
            raise DerivedDihedralUnifiedCompositingError(
                "unified paint relation references an unknown item"
            )
        if relation.near_item_id not in outgoing[relation.far_item_id]:
            outgoing[relation.far_item_id].add(relation.near_item_id)
            indegree[relation.near_item_id] += 1
    ready = [item_id for item_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    result: list[str] = []
    while ready:
        current = heapq.heappop(ready)
        result.append(current)
        for target in sorted(outgoing[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)
    if len(result) != len(identities):
        cyclic = sorted(
            item_id for item_id, degree in indegree.items() if degree > 0
        )
        raise DerivedDihedralUnifiedCompositingError(
            "unified face/stroke painter order contains a cycle: "
            + ", ".join(cyclic)
        )
    return tuple(result)


def _isometric_projection() -> np.ndarray:
    view = np.asarray((1.0, 1.0, 1.0), dtype=float)
    view /= np.linalg.norm(view)
    right = np.cross(np.asarray((0.0, 0.0, 1.0)), view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    up /= np.linalg.norm(up)
    return np.asarray((right, up, view), dtype=float)


class DerivedDihedralUnifiedSortMigrationTests(unittest.TestCase):
    def test_random_dags_preserve_the_legacy_lexicographic_order(self) -> None:
        rng = random.Random(0xD1ED6A1)
        for _case in range(2500):
            count = rng.randint(0, 14)
            nodes = tuple(f"item:{index:02d}" for index in range(count))
            shuffled = list(nodes)
            rng.shuffle(shuffled)
            if shuffled and rng.random() < 0.2:
                shuffled.append(rng.choice(shuffled))

            relations: list[UnifiedPaintRelation] = []
            for first_index, first in enumerate(nodes):
                for second in nodes[first_index + 1 :]:
                    if rng.random() < 0.19:
                        relations.append(_relation(first, second))
                        if rng.random() < 0.1:
                            relations.append(_relation(first, second))
            rng.shuffle(relations)
            relation_tuple = tuple(relations)
            item_tuple = tuple(shuffled)
            self.assertEqual(
                _draw_order(item_tuple, relation_tuple),
                _legacy_draw_order(item_tuple, relation_tuple),
            )

    def test_unknown_relation_endpoint_remains_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            DerivedDihedralUnifiedCompositingError,
            "unknown item",
        ):
            _draw_order(("face:a",), (_relation("face:a", "missing"),))

    def test_self_relation_retains_the_historical_cycle_error(self) -> None:
        with self.assertRaisesRegex(
            DerivedDihedralUnifiedCompositingError,
            r"contains a cycle: face:a$",
        ):
            _draw_order(("face:a",), (_relation("face:a", "face:a"),))

    def test_multi_node_cycle_is_adapted_to_the_domain_error(self) -> None:
        with self.assertRaisesRegex(
            DerivedDihedralUnifiedCompositingError,
            r"contains a cycle: face:a, face:b, stroke:c$",
        ):
            _draw_order(
                ("stroke:c", "face:b", "face:a"),
                (
                    _relation("face:a", "face:b"),
                    _relation("face:b", "stroke:c"),
                    _relation("stroke:c", "face:a"),
                ),
            )

    def test_public_compositing_path_reaches_the_shared_sorter(self) -> None:
        model = DerivedDihedralModel.from_solid(
            "cube-shared-compositor-reachability",
            cube_model(),
            entity_id="copy",
            source_face_ids=("front", "top"),
        )
        target = (
            "polyhedron_visibility.dihedral_extraction.unified_compositing."
            "stable_topological_sort"
        )
        with patch(target, wraps=stable_topological_sort) as shared_sort:
            frame = compute_derived_dihedral_unified_compositing(
                model,
                transform=RigidTransform3D.translation_by((0.0, 0.0, -0.5)),
                projection_matrix=_isometric_projection(),
            )
        self.assertGreater(shared_sort.call_count, 0)
        self.assertEqual(set(frame.draw_order), set(frame.item_ids))

    def test_representative_motion_frames_match_the_legacy_sorter(self) -> None:
        model = DerivedDihedralModel.from_solid(
            "cube-shared-compositor-differential",
            cube_model(),
            entity_id="copy",
            source_face_ids=("front", "top"),
        )
        for shift in np.linspace(0.0, -3.0, 31):
            with self.subTest(shift=float(shift)):
                frame = compute_derived_dihedral_unified_compositing(
                    model,
                    transform=RigidTransform3D.translation_by(
                        (0.0, 0.0, float(shift))
                    ),
                    projection_matrix=_isometric_projection(),
                )
                self.assertEqual(
                    frame.draw_order,
                    _legacy_draw_order(frame.item_ids, frame.order_relations),
                )


if __name__ == "__main__":
    unittest.main()
