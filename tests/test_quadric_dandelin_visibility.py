from __future__ import annotations

from dataclasses import replace
from math import pi
import unittest

import numpy as np

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics import (
    ConeModel,
    ConeSpec,
    DandelinTangentContactEvidence,
    DandelinVisibilityError,
    SectionPlane,
    canonical_dandelin_visibility_json,
    compute_dandelin_construction,
    compute_dandelin_visibility_frame,
    fit_plane_display_patch,
)
from polyhedron_visibility.visibility import VisibilityKind


VIEW = ParallelView.from_matrix(
    np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 0.8, 0.6),
            (0.0, -0.6, 0.8),
        ),
        dtype=float,
    )
)
ROTATED_VIEW = ParallelView.from_matrix(
    np.asarray(
        (
            (0.8, -0.6, 0.0),
            (0.3, 0.4, -0.8660254037844386),
            (0.5196152422706632, 0.6928203230275509, 0.5),
        ),
        dtype=float,
    )
)


def _ellipse():
    cone = ConeSpec(
        "cone",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        pi / 6.0,
        (0.0, 9.0),
        radial_axis=(1.0, 0.0, 0.0),
        model=ConeModel.OPEN_SINGLE,
    )
    plane = SectionPlane(
        "cut",
        (0.0, 0.0, 2.0),
        (0.6, 0.0, 0.8),
        u_axis=(0.0, 1.0, 0.0),
    )
    construction = compute_dandelin_construction("dan", cone, plane)
    patch = fit_plane_display_patch(
        "patch",
        plane,
        cone.render_components,
        margin_ratio=0.14,
    ).patch
    return construction, patch


class DandelinVisibilityTests(unittest.TestCase):
    def test_existing_quadric_kernel_certifies_hidden_line_frame(self) -> None:
        construction, patch = _ellipse()
        frame = compute_dandelin_visibility_frame(
            construction,
            VIEW,
            directrix_patch=patch,
        )

        self.assertTrue(frame.curve_visibility_authoritative)
        self.assertFalse(frame.surface_visibility_authoritative)
        self.assertEqual(len(frame.tangent_contacts), len(construction.spheres))
        self.assertGreater(frame.hidden_span_count, 0)
        self.assertEqual(
            {item.role for item in frame.strokes},
            {
                "cone_boundary",
                "sphere_silhouette",
                "contact_circle",
                "section_curve",
                "directrix",
            },
        )
        self.assertTrue(
            any(
                span.kind is VisibilityKind.VISIBLE
                for stroke in frame.strokes
                for span in stroke.spans
            )
        )
        self.assertTrue(
            any(
                span.kind is VisibilityKind.HIDDEN
                for stroke in frame.strokes
                for span in stroke.spans
            )
        )
        self.assertEqual(
            canonical_dandelin_visibility_json(frame),
            canonical_dandelin_visibility_json(
                compute_dandelin_visibility_frame(
                    construction,
                    VIEW,
                    directrix_patch=patch,
                )
            ),
        )

    def test_sphere_silhouette_excludes_its_owner_but_keeps_external_occluders(
        self,
    ) -> None:
        construction, patch = _ellipse()
        frame = compute_dandelin_visibility_frame(
            construction,
            VIEW,
            directrix_patch=patch,
        )

        silhouettes = tuple(
            item for item in frame.strokes if item.role == "sphere_silhouette"
        )
        self.assertEqual(len(silhouettes), len(construction.spheres))
        self.assertTrue(any(item.hidden_span_count for item in silhouettes))
        for stroke in silhouettes:
            for span in stroke.spans:
                self.assertNotIn(stroke.source_ref, span.occluder_surface_ids)

    def test_camera_rotation_recomputes_visibility_without_changing_source_ids(
        self,
    ) -> None:
        construction, patch = _ellipse()
        first = compute_dandelin_visibility_frame(
            construction,
            VIEW,
            directrix_patch=patch,
        )
        second = compute_dandelin_visibility_frame(
            construction,
            ROTATED_VIEW,
            directrix_patch=patch,
        )

        self.assertEqual(
            tuple(item.source_id for item in first.strokes),
            tuple(item.source_id for item in second.strokes),
        )
        self.assertNotEqual(
            tuple(
                tuple(
                    (span.kind, span.occluder_surface_ids)
                    for span in item.spans
                )
                for item in first.strokes
            ),
            tuple(
                tuple(
                    (span.kind, span.occluder_surface_ids)
                    for span in item.spans
                )
                for item in second.strokes
            ),
        )

    def test_open_double_contacts_identify_their_exact_nappes(self) -> None:
        cone = ConeSpec(
            "double",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (-4.0, 4.0),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.OPEN_DOUBLE,
        )
        plane = SectionPlane(
            "hyperbola-plane",
            (0.0, 0.0, 2.0),
            ((1.0 - 0.2**2) ** 0.5, 0.0, 0.2),
            u_axis=(0.0, 1.0, 0.0),
        )
        construction = compute_dandelin_construction("hyperbola", cone, plane)
        patch = fit_plane_display_patch(
            "hyperbola-patch",
            plane,
            cone.render_components,
            margin_ratio=0.14,
        ).patch

        frame = compute_dandelin_visibility_frame(
            construction,
            VIEW,
            directrix_patch=patch,
        )

        self.assertEqual(
            {item.cone_surface_id for item in frame.tangent_contacts},
            {"double:nappe:negative", "double:nappe:positive"},
        )
        self.assertEqual(
            {item.sphere_id for item in frame.tangent_contacts},
            {item.sphere_id for item in construction.spheres},
        )

    def test_tangent_evidence_and_directrix_boundary_fail_closed(self) -> None:
        construction, _patch = _ellipse()
        with self.assertRaisesRegex(
            DandelinVisibilityError,
            "directrix visibility requires.*finite display patch",
        ):
            compute_dandelin_visibility_frame(construction, VIEW)
        with self.assertRaisesRegex(
            DandelinVisibilityError,
            "equal-depth",
        ):
            replace(
                DandelinTangentContactEvidence("sphere", "cone", "circle"),
                equal_depth_contact=False,
            )


if __name__ == "__main__":
    unittest.main()
