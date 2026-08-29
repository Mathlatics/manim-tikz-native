from __future__ import annotations

import unittest
from unittest.mock import patch

from manim import Scene, VGroup, VMobject

from polyhedron_visibility.painter_band import ManagedPainterBand


def _stroke(x: float) -> VMobject:
    return VMobject().set_points_as_corners(((x, 0.0, 0.0), (x, 1.0, 0.0)))


class IncrementalPainterBandTests(unittest.TestCase):
    def test_only_items_with_changed_ranks_receive_family_writes(self) -> None:
        scene = Scene()
        first, second, third = (_stroke(value) for value in (0.0, 1.0, 2.0))
        root = VGroup(first, second, third)
        scene.add(root)
        band = ManagedPainterBand(
            z_band=(20.0, 30.0),
            managed_roots=(root,),
        )
        band.configure(
            containers=(scene.mobjects,),
            sources={"root": root},
        )
        initial = band.prepare(
            draw_order=("first", "second", "third"),
            item_mobjects={
                "first": first,
                "second": second,
                "third": third,
            },
        )
        band.apply(initial)

        reordered = band.prepare(
            draw_order=("first", "third", "second"),
            item_mobjects={
                "first": first,
                "second": second,
                "third": third,
            },
        )
        self.assertEqual(
            tuple(item.item_id for item in band.changed_items(reordered)),
            ("third", "second"),
        )
        with (
            patch.object(first, "set_z_index", wraps=first.set_z_index) as first_set,
            patch.object(second, "set_z_index", wraps=second.set_z_index) as second_set,
            patch.object(third, "set_z_index", wraps=third.set_z_index) as third_set,
        ):
            band.apply(reordered)

        first_set.assert_not_called()
        second_set.assert_called_once_with(30.0, family=True)
        third_set.assert_called_once_with(25.0, family=True)
        self.assertEqual(
            band.active_z_indices,
            {"first": 20.0, "third": 25.0, "second": 30.0},
        )

    def test_unchanged_band_has_no_mutation_targets(self) -> None:
        scene = Scene()
        first, second = (_stroke(value) for value in (0.0, 1.0))
        root = VGroup(first, second)
        scene.add(root)
        band = ManagedPainterBand(
            z_band=(4.0, 8.0),
            managed_roots=(root,),
        )
        band.configure(containers=(scene.mobjects,), sources={"root": root})
        prepared = band.prepare(
            draw_order=("first", "second"),
            item_mobjects={"first": first, "second": second},
        )
        band.apply(prepared)
        self.assertEqual(band.changed_items(prepared), ())
        with (
            patch.object(first, "set_z_index", wraps=first.set_z_index) as first_set,
            patch.object(second, "set_z_index", wraps=second.set_z_index) as second_set,
        ):
            band.apply(prepared)
        first_set.assert_not_called()
        second_set.assert_not_called()


if __name__ == "__main__":
    unittest.main()
