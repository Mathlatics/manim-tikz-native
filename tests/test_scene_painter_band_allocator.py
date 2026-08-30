from __future__ import annotations

from types import SimpleNamespace
import unittest

from polyhedron_visibility.painter_band import (
    ScenePainterBandError,
    ScenePainterBandReservation,
    release_scene_painter_band,
    reserve_scene_painter_band,
    scene_painter_band_allocations,
)


class ScenePainterBandAllocatorTests(unittest.TestCase):
    def test_preferred_band_uses_first_available_slot_above_conflicts(self) -> None:
        scene = SimpleNamespace()
        first = ScenePainterBandReservation(("quadric", "first"), (20.0, 30.0))
        second = ScenePainterBandReservation(("quadric", "second"), (20.0, 30.0))
        third = ScenePainterBandReservation(("quadric", "third"), (25.0, 27.0))

        self.assertEqual(reserve_scene_painter_band(scene, first), (20.0, 30.0))
        self.assertEqual(reserve_scene_painter_band(scene, second), (31.0, 41.0))
        self.assertEqual(reserve_scene_painter_band(scene, third), (42.0, 44.0))
        bands = tuple(
            allocation.z_band
            for allocation in scene_painter_band_allocations(scene)
        )
        for index, band in enumerate(bands):
            for other in bands[index + 1 :]:
                self.assertTrue(band[1] < other[0] or other[1] < band[0])

    def test_same_token_is_idempotent_but_duplicate_owner_is_rejected(self) -> None:
        scene = SimpleNamespace()
        owner = ("quadric", "section-a")
        reservation = ScenePainterBandReservation(owner, (20.0, 30.0))
        duplicate = ScenePainterBandReservation(owner, (40.0, 50.0))

        first = reserve_scene_painter_band(scene, reservation)
        self.assertEqual(reserve_scene_painter_band(scene, reservation), first)
        before = scene_painter_band_allocations(scene)
        with self.assertRaisesRegex(ScenePainterBandError, "duplicate.*owner"):
            reserve_scene_painter_band(scene, duplicate)
        self.assertEqual(scene_painter_band_allocations(scene), before)

    def test_exact_conflict_fails_without_mutating_active_allocations(self) -> None:
        scene = SimpleNamespace()
        automatic = ScenePainterBandReservation("automatic", (20.0, 30.0))
        reserve_scene_painter_band(scene, automatic)
        before = scene_painter_band_allocations(scene)
        exact = ScenePainterBandReservation(
            "exact",
            (30.0, 40.0),
            exact=True,
        )

        with self.assertRaisesRegex(ScenePainterBandError, "exact.*conflicts"):
            reserve_scene_painter_band(scene, exact)
        self.assertEqual(scene_painter_band_allocations(scene), before)

        nonconflicting = ScenePainterBandReservation(
            "nonconflicting-exact",
            (40.0, 50.0),
            exact=True,
        )
        self.assertEqual(
            reserve_scene_painter_band(scene, nonconflicting),
            (40.0, 50.0),
        )

    def test_release_requires_matching_token_and_cleans_up_registry(self) -> None:
        scene = SimpleNamespace()
        owner = ("quadric", "section-a")
        reservation = ScenePainterBandReservation(owner, (20.0, 30.0))
        foreign = ScenePainterBandReservation(owner, (20.0, 30.0))
        reserve_scene_painter_band(scene, reservation)

        with self.assertRaisesRegex(ScenePainterBandError, "different.*token"):
            release_scene_painter_band(scene, foreign)
        self.assertEqual(len(scene_painter_band_allocations(scene)), 1)
        self.assertTrue(release_scene_painter_band(scene, reservation))
        self.assertFalse(release_scene_painter_band(scene, reservation))
        self.assertEqual(scene_painter_band_allocations(scene), ())
        self.assertFalse(
            hasattr(
                scene,
                "_polyhedron_visibility_scene_painter_band_reservations",
            )
        )
        self.assertEqual(
            reserve_scene_painter_band(scene, reservation),
            (20.0, 30.0),
        )

    def test_release_makes_preferred_band_reusable(self) -> None:
        scene = SimpleNamespace()
        first = ScenePainterBandReservation("first", (20.0, 30.0))
        second = ScenePainterBandReservation("second", (20.0, 30.0))
        reserve_scene_painter_band(scene, first)
        self.assertEqual(reserve_scene_painter_band(scene, second), (31.0, 41.0))
        release_scene_painter_band(scene, first)
        third = ScenePainterBandReservation("third", (20.0, 30.0))
        self.assertEqual(reserve_scene_painter_band(scene, third), (20.0, 30.0))

    def test_request_validation_happens_before_scene_mutation(self) -> None:
        scene = SimpleNamespace()
        with self.assertRaisesRegex(ScenePainterBandError, "two-value tuple"):
            ScenePainterBandReservation("bad", [0.0, 1.0])  # type: ignore[arg-type]
        with self.assertRaisesRegex(ScenePainterBandError, "hashable"):
            ScenePainterBandReservation([], (0.0, 1.0))  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "exact must be a bool"):
            ScenePainterBandReservation(
                "bad-exact",
                (0.0, 1.0),
                exact=1,  # type: ignore[arg-type]
            )
        self.assertEqual(scene_painter_band_allocations(scene), ())


if __name__ == "__main__":
    unittest.main()
