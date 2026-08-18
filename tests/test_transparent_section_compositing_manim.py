from __future__ import annotations

import unittest

import numpy as np
from manim import Polygon, Scene, VGroup, tempconfig

from polyhedron_visibility import VisibilityModel
from polyhedron_visibility.sections.compositing import (
    compute_transparent_section_compositing,
)
from polyhedron_visibility.sections.compositing_manim import (
    TransparentSectionLayer,
    TransparentSectionManimError,
    transparent_triangle_capacity,
)
from polyhedron_visibility.sections.contract import SectionPlane3D
from polyhedron_visibility.contract import TolerancePolicy


_VERTICES = {
    "A": (-1.0, -1.0, -1.0),
    "B": (1.0, -1.0, -1.0),
    "C": (1.0, 1.0, -1.0),
    "D": (-1.0, 1.0, -1.0),
    "E": (-1.0, -1.0, 1.0),
    "F": (1.0, -1.0, 1.0),
    "G": (1.0, 1.0, 1.0),
    "H": (-1.0, 1.0, 1.0),
}

_FACES = {
    "back": ("A", "D", "C", "B"),
    "front": ("E", "F", "G", "H"),
    "bottom": ("A", "B", "F", "E"),
    "right": ("B", "C", "G", "F"),
    "top": ("D", "H", "G", "C"),
    "left": ("A", "E", "H", "D"),
}


def _model() -> VisibilityModel:
    return VisibilityModel.from_dict(
        {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": "transparent-cube",
            "vertices": [
                {"vertexId": key, "entryPosition": value}
                for key, value in _VERTICES.items()
            ],
            "faces": [
                {"faceId": key, "vertexIds": list(value)}
                for key, value in _FACES.items()
            ],
            "strokes": [],
        }
    )


def _plane(offset: float = 0.0) -> SectionPlane3D:
    return SectionPlane3D(
        "cut",
        (offset / 3.0, offset / 3.0, offset / 3.0),
        (1.0, 1.0, 1.0),
        3.0,
        3.0,
        u_axis=(1.0, -1.0, 0.0),
    )


class TransparentSectionLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tempconfig({"renderer": "cairo", "frame_rate": 12})
        self.config.__enter__()

    def tearDown(self) -> None:
        self.config.__exit__(None, None, None)

    def _fixture(self) -> tuple[Scene, VisibilityModel, dict[str, Polygon]]:
        scene = Scene()
        model = _model()
        faces: dict[str, Polygon] = {}
        for index, face in enumerate(model.faces):
            polygon = Polygon(
                *[_VERTICES[item] for item in face.vertex_ids],
                fill_color=("#7BA3D8" if index % 2 else "#8CC8C0"),
                fill_opacity=0.24,
                stroke_opacity=0.0,
            )
            polygon.set_z_index(2.0 + index)
            faces[face.face_id] = polygon
        scene.add(VGroup(*faces.values()))
        return scene, model, faces

    def test_fixed_triangle_pool_applies_two_local_orders_and_restores(self) -> None:
        scene, model, faces = self._fixture()
        layer = TransparentSectionLayer(
            model,
            faces,
            tolerance_policy=TolerancePolicy(),
            source_coordinate_mode="world",
        )
        scene.add(layer.root)
        frame = compute_transparent_section_compositing(
            "section",
            model,
            _plane(),
            projection_matrix=np.eye(3),
        )
        prepared = layer.prepare(
            frame,
            world_points={key: np.asarray(value, dtype=float) for key, value in _VERTICES.items()},
            display_point_provider=lambda point: point,
            plane_fill_color="#8CC8C0",
            plane_fill_opacity=0.15,
            section_fill_color="#F4C95D",
            section_fill_opacity=0.42,
            face_depth_cue=None,
            containers=(scene.mobjects, scene.foreground_mobjects),
        )
        identities = layer.identities()
        self.assertEqual(layer.capacity, transparent_triangle_capacity(model))
        layer.capture_and_hide()
        layer.apply(prepared)
        self.assertEqual(set(layer.active_fragment_ids), set(frame.fragment_map))
        self.assertTrue(
            all(float(item.get_fill_opacity()) == 0.0 for item in faces.values())
        )
        active_slots = [
            item for item in layer.slots if float(item.get_fill_opacity()) > 0.0
        ]
        self.assertEqual(len(active_slots), len(frame.fragments))
        self.assertGreater(len({float(item.z_index) for item in active_slots}), 1)

        next_frame = compute_transparent_section_compositing(
            "section",
            model,
            _plane(2.0),
            projection_matrix=np.eye(3),
        )
        next_prepared = layer.prepare(
            next_frame,
            world_points={key: np.asarray(value, dtype=float) for key, value in _VERTICES.items()},
            display_point_provider=lambda point: point,
            plane_fill_color="#8CC8C0",
            plane_fill_opacity=0.15,
            section_fill_color="#F4C95D",
            section_fill_opacity=0.42,
            face_depth_cue=None,
            containers=(scene.mobjects, scene.foreground_mobjects),
        )
        layer.apply(next_prepared)
        self.assertEqual(layer.identities(), identities)
        self.assertEqual(set(layer.active_fragment_ids), set(next_frame.fragment_map))

        layer.restore()
        self.assertTrue(
            all(abs(float(item.get_fill_opacity()) - 0.24) <= 1.0e-12 for item in faces.values())
        )

    def test_unrelated_drawable_in_managed_z_band_fails_before_hiding(self) -> None:
        scene, model, faces = self._fixture()
        unrelated = Polygon(
            (-3, -3, 0), (-2, -3, 0), (-2, -2, 0),
            fill_opacity=0.2,
            stroke_opacity=0.0,
        ).set_z_index(4.5)
        scene.add(unrelated)
        layer = TransparentSectionLayer(
            model,
            faces,
            tolerance_policy=TolerancePolicy(),
            source_coordinate_mode="world",
        )
        frame = compute_transparent_section_compositing(
            "section", model, _plane(), projection_matrix=np.eye(3)
        )
        with self.assertRaisesRegex(
            TransparentSectionManimError, "unrelated Scene drawable"
        ):
            layer.prepare(
                frame,
                world_points={
                    key: np.asarray(value, dtype=float)
                    for key, value in _VERTICES.items()
                },
                display_point_provider=lambda point: point,
                plane_fill_color="#8CC8C0",
                plane_fill_opacity=0.15,
                section_fill_color="#F4C95D",
                section_fill_opacity=0.42,
                face_depth_cue=None,
                containers=(scene.mobjects, scene.foreground_mobjects),
            )
        self.assertTrue(
            all(abs(float(item.get_fill_opacity()) - 0.24) <= 1.0e-12 for item in faces.values())
        )

    def test_source_gradient_is_rejected(self) -> None:
        scene, model, faces = self._fixture()
        source = faces["back"]
        source.fill_rgbas = np.asarray(
            ((0.2, 0.3, 0.4, 0.2), (0.7, 0.8, 0.9, 0.2)), dtype=float
        )
        layer = TransparentSectionLayer(
            model,
            faces,
            tolerance_policy=TolerancePolicy(),
            source_coordinate_mode="world",
        )
        frame = compute_transparent_section_compositing(
            "section", model, _plane(), projection_matrix=np.eye(3)
        )
        with self.assertRaisesRegex(
            TransparentSectionManimError, "non-gradient"
        ):
            layer.prepare(
                frame,
                world_points={
                    key: np.asarray(value, dtype=float)
                    for key, value in _VERTICES.items()
                },
                display_point_provider=lambda point: point,
                plane_fill_color="#8CC8C0",
                plane_fill_opacity=0.15,
                section_fill_color="#F4C95D",
                section_fill_opacity=0.42,
                face_depth_cue=None,
                containers=(scene.mobjects, scene.foreground_mobjects),
            )


if __name__ == "__main__":
    unittest.main()
