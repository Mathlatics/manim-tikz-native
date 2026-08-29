from __future__ import annotations

from math import pi
import unittest
from unittest.mock import patch

from manim import Scene, config

from polyhedron_visibility.quadrics import (
    ConeModel,
    ConeSpec,
    DandelinSection3D,
    DandelinSectionAuthoringError,
    QuadricManimLimits,
    SectionPlane,
)
from polyhedron_visibility.quadrics.composite_authoring import (
    CompositeQuadricSection3D,
)
from polyhedron_visibility.quadrics.boundary_compositing import (
    BoundarySourceKind,
)
from polyhedron_visibility.visibility import VisibilityKind


def _limits() -> QuadricManimLimits:
    return QuadricManimLimits(
        max_surfaces=4,
        max_curves=8,
        max_fragments_per_curve=20,
        max_segments_per_fragment=384,
        max_surface_segments=512,
        max_dashes_per_fragment=96,
        max_projected_length=32.0,
        max_total_mobjects=10000,
        max_boundary_sources=32,
    )


def _ellipse_facade(scene: Scene) -> DandelinSection3D:
    return DandelinSection3D(
        scene,
        cone=ConeSpec(
            "dandelin-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 9.0),
            model=ConeModel.OPEN_SINGLE,
        ),
        plane=SectionPlane(
            "dandelin-plane",
            (0.0, 0.0, 2.0),
            (0.6, 0.0, 0.8),
            u_axis=(0.0, 1.0, 0.0),
        ),
        construction_id="dandelin-authoring",
        limits=_limits(),
        max_chord_error=0.02,
        section_max_screen_error=0.12,
    )


def _hyperbola_facade(scene: Scene) -> DandelinSection3D:
    return DandelinSection3D(
        scene,
        cone=ConeSpec(
            "double-cone",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (-4.0, 4.0),
            model=ConeModel.OPEN_DOUBLE,
        ),
        plane=SectionPlane(
            "hyperbola-plane",
            (0.0, 0.0, 2.0),
            ((1.0 - 0.2**2) ** 0.5, 0.0, 0.2),
            u_axis=(0.0, 1.0, 0.0),
        ),
        construction_id="hyperbola-authoring",
        limits=_limits(),
        max_chord_error=0.02,
        section_max_screen_error=0.12,
    )


class DandelinSection3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config.renderer = "cairo"

    def test_static_facade_keeps_fixed_identity_and_restores_scene(self) -> None:
        scene = Scene()
        facade = _ellipse_facade(scene)
        identities = facade.slot_identities()
        display_identity = id(facade.display_mobject)

        facade.attach()

        self.assertTrue(facade.attached)
        self.assertEqual(facade.slot_identities(), identities)
        self.assertEqual(id(facade.display_mobject), display_identity)
        self.assertFalse(facade.visibility_authoritative)
        self.assertEqual(facade.overlay_mode, "diagrammatic")
        self.assertGreater(len(scene.mobjects), 0)

        facade.restore()

        self.assertFalse(facade.attached)
        self.assertEqual(scene.mobjects, [])
        self.assertEqual(facade.slot_identities(), identities)

    def test_attach_failure_rolls_back_the_already_attached_section(self) -> None:
        scene = Scene()
        facade = _ellipse_facade(scene)

        with patch.object(
            facade.overlay_controller,
            "attach",
            side_effect=RuntimeError("injected overlay failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected overlay failure"):
                facade.attach()

        self.assertFalse(facade.attached)
        self.assertFalse(facade.section_controller.attached)
        self.assertFalse(facade.overlay_controller.attached)
        self.assertEqual(scene.mobjects, [])

    def test_projection_callback_is_rejected_before_allocating_dynamic_overlay(self) -> None:
        scene = Scene()
        with self.assertRaisesRegex(
            DandelinSectionAuthoringError,
            "immutable parallel projection",
        ):
            DandelinSection3D(
                scene,
                cone=ConeSpec(
                    "callback-cone",
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    pi / 6.0,
                    (0.0, 9.0),
                    model=ConeModel.OPEN_SINGLE,
                ),
                plane=SectionPlane(
                    "callback-plane",
                    (0.0, 0.0, 2.0),
                    (0.6, 0.0, 0.8),
                    u_axis=(0.0, 1.0, 0.0),
                ),
                construction_id="callback-dandelin",
                projection=lambda _scene: ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            )

    def test_circular_section_attaches_two_spheres_without_fake_directrices(
        self,
    ) -> None:
        scene = Scene()
        facade = DandelinSection3D(
            scene,
            cone=ConeSpec(
                "circle-cone",
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                pi / 6.0,
                (0.0, 10.0),
                model=ConeModel.OPEN_SINGLE,
            ),
            plane=SectionPlane(
                "circle-plane",
                (0.0, 0.0, 2.0),
                (0.0, 0.0, 1.0),
                u_axis=(1.0, 0.0, 0.0),
            ),
            construction_id="circle-authoring",
            limits=_limits(),
            max_chord_error=0.02,
            section_max_screen_error=0.12,
        )
        identities = facade.slot_identities()

        facade.attach()
        try:
            self.assertEqual(facade.construction.supporting_kind.value, "circle")
            self.assertEqual(len(facade.construction.spheres), 2)
            self.assertEqual(facade.construction.directrices, ())
            self.assertEqual(facade.slot_identities(), identities)
        finally:
            facade.restore()

        self.assertEqual(scene.mobjects, [])

    def test_open_double_hyperbola_reuses_the_existing_composite_controller(self) -> None:
        scene = Scene()
        facade = _hyperbola_facade(scene)

        self.assertIsInstance(
            facade.section_controller,
            CompositeQuadricSection3D,
        )
        self.assertEqual(
            {item.nappe_sign for item in facade.construction.spheres},
            {-1, 1},
        )
        identities = facade.slot_identities()

        facade.attach()
        try:
            self.assertEqual(facade.slot_identities(), identities)
            boundary = facade.section_controller.last_boundary_frame
            self.assertIsNotNone(boundary)
            assert boundary is not None
            sources = {item.source_id: item for item in boundary.sources}
            hidden_plane_edges = tuple(
                item
                for item in boundary.fragments
                if sources[item.source_id].source_kind
                is BoundarySourceKind.PLANE_PATCH_EDGE
                and item.effective_visibility_kind is VisibilityKind.HIDDEN
            )
            self.assertTrue(hidden_plane_edges)
            child_ids = {
                item.surface_id
                for item in facade.section_controller.children
            }
            self.assertTrue(
                all(
                    len(item.occluder_surface_ids) == 1
                    and item.occluder_surface_ids[0] in child_ids
                    for item in hidden_plane_edges
                )
            )
            self.assertEqual(
                {
                    item.occluder_surface_ids[0]
                    for item in hidden_plane_edges
                },
                child_ids,
            )
        finally:
            facade.restore()

        self.assertEqual(scene.mobjects, [])
        self.assertEqual(facade.slot_identities(), identities)

    def test_open_double_overlay_failure_restores_both_controller_layers(self) -> None:
        scene = Scene()
        facade = _hyperbola_facade(scene)
        identities = facade.slot_identities()

        with patch.object(
            facade.overlay_controller,
            "attach",
            side_effect=RuntimeError("injected double overlay failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "injected double overlay failure",
            ):
                facade.attach()

        self.assertFalse(facade.attached)
        self.assertFalse(facade.section_controller.attached)
        self.assertFalse(facade.overlay_controller.attached)
        self.assertEqual(scene.mobjects, [])
        self.assertEqual(facade.slot_identities(), identities)


if __name__ == "__main__":
    unittest.main()
