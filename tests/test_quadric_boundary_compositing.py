from __future__ import annotations

from math import pi, sqrt
import unittest

import numpy as np

from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.quadrics.boundary_compositing import (
    BoundaryOcclusionScope,
    BoundaryRenderIntent,
    BoundarySectionAnchors,
    BoundarySemanticKind,
    BoundarySourceKind,
    QuadricBoundaryVisibilitySpan,
    canonical_quadric_boundary_compositing_json,
    compute_boundary_visibility,
    compute_quadric_boundary_compositing,
)
from polyhedron_visibility.quadrics.compositing import (
    QuadricPaintPolicy,
    QuadricPaintRelation,
)
from polyhedron_visibility.quadrics.contract import (
    ConeSpec,
    CylinderSpec,
    PlaneDisplayPatchSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.curve_intersections import (
    compute_projected_curve_crossings,
)
from polyhedron_visibility.quadrics.curves import SegmentCurve
from polyhedron_visibility.quadrics.surface_boundaries import (
    GeneratorBoundarySpec,
    build_surface_boundary_sources,
    curve_boundary_source,
    plane_outline_sources,
    surface_boundary_source_ids,
)
from polyhedron_visibility.topology import ParameterInterval
from polyhedron_visibility.visibility import VisibilityKind


VIEW = ParallelView.from_matrix(
    (
        (-1.0 / sqrt(2.0), 1.0 / sqrt(2.0), 0.0),
        (-1.0 / sqrt(6.0), -1.0 / sqrt(6.0), 2.0 / sqrt(6.0)),
        (1.0 / sqrt(3.0), 1.0 / sqrt(3.0), 1.0 / sqrt(3.0)),
    )
)


IDENTITY_VIEW = ParallelView.from_matrix(np.eye(3))

class QuadricBoundaryContractTests(unittest.TestCase):
    def test_depth_aware_visibility_uses_source_scope(self) -> None:
        owner = SphereSpec("owner", (0.0, 0.0, 0.0), 1.0)
        other = SphereSpec("other", (0.0, 0.0, 3.0), 0.4)
        silhouette = build_surface_boundary_sources(
            (owner,), VIEW, include_cap_rims=False
        )[0]
        self.assertIs(
            silhouette.occlusion_scope,
            BoundaryOcclusionScope.EXTERNAL_ONLY,
        )
        spans = compute_boundary_visibility(
            (silhouette,), (owner, other), VIEW
        )[silhouette.source_id]
        self.assertTrue(spans)
        self.assertTrue(
            all("owner" not in span.occluder_surface_ids for span in spans)
        )

    def test_surface_boundary_sources_cover_caps_and_silhouettes(self) -> None:
        cylinder = CylinderSpec(
            "cylinder",
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 1.0),
            1.0,
            (0.0, 2.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        cone = ConeSpec(
            "cone",
            (0.0, 0.0, -2.0),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 4.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        ids = surface_boundary_source_ids((cylinder, cone))
        self.assertIn("boundary:cylinder:cap_min:rim", ids)
        self.assertIn("boundary:cylinder:cap_max:rim", ids)
        self.assertIn("boundary:cone:cap_max:rim", ids)
        self.assertNotIn("boundary:cone:cap_min:rim", ids)
        self.assertIn("boundary:cylinder:silhouette:generator:0", ids)
        sources = build_surface_boundary_sources((cylinder, cone), VIEW)
        source_ids = {item.source_id for item in sources}
        self.assertTrue(source_ids <= set(ids))
        self.assertTrue(
            any(
                item.semantic_kind is BoundarySemanticKind.SURFACE_BOUNDARY
                for item in sources
            )
        )
        self.assertTrue(
            any(
                item.semantic_kind is BoundarySemanticKind.TRUE_SILHOUETTE
                for item in sources
            )
        )

    def test_explicit_generator_is_owner_aware(self) -> None:
        cone = ConeSpec(
            "cone",
            (0.0, 0.0, -2.0),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 4.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        source = build_surface_boundary_sources(
            (cone,),
            VIEW,
            (GeneratorBoundarySpec("generator:a", "cone", 0.4),),
            include_cap_rims=False,
            include_silhouettes=False,
        )[0]
        self.assertEqual(source.owner_surface_id, "cone")
        self.assertIs(
            source.occlusion_scope,
            BoundaryOcclusionScope.OWNER_AND_EXTERNAL,
        )
        self.assertIs(source.source_kind, BoundarySourceKind.SURFACE_GENERATOR)

    def test_plane_outline_policy_matrix(self) -> None:
        plane = SectionPlane(
            "cut", (0.0, 0.0, 0.0), (0.7, 0.0, 1.0), u_axis=(0.0, 1.0, 0.0)
        )
        patch = PlaneDisplayPatchSpec("patch", "cut", 2.0, 1.5)
        source = plane_outline_sources(plane, patch)[0]
        spans = {
            source.source_id: (
                QuadricBoundaryVisibilitySpan(
                    ParameterInterval(0.0, 0.25),
                    VisibilityKind.VISIBLE,
                    depth_role="outside_projection",
                ),
                QuadricBoundaryVisibilitySpan(
                    ParameterInterval(0.25, 0.5),
                    VisibilityKind.HIDDEN,
                    ("solid",),
                    "behind_surface",
                ),
                QuadricBoundaryVisibilitySpan(
                    ParameterInterval(0.5, 0.75),
                    VisibilityKind.HIDDEN,
                    ("solid",),
                    "between_surface_sheets",
                ),
                QuadricBoundaryVisibilitySpan(
                    ParameterInterval(0.75, 1.0),
                    VisibilityKind.VISIBLE,
                    depth_role="in_front_of_surface",
                ),
            )
        }
        anchors = BoundarySectionAnchors(
            "plane-behind",
            "outline-behind",
            "surface-back",
            "plane-outside",
            "outline-outside",
            "plane-between",
            "outline-between",
            "surface-front",
            "plane-front",
            "outline-front",
        )
        parent = (
            "plane-behind",
            "outline-behind",
            "surface-back",
            "plane-outside",
            "outline-outside",
            "plane-between",
            "outline-between",
            "surface-front",
            "plane-front",
            "outline-front",
        )
        parent_relations = tuple(
            QuadricPaintRelation(left, right, "parent")
            for left, right in zip(parent, parent[1:])
        )
        physical = compute_quadric_boundary_compositing(
            (source,),
            spans,
            paint_policy=QuadricPaintPolicy.PHYSICAL,
            parent_item_ids=parent,
            parent_relations=parent_relations,
            surface_item_by_id={"solid": "surface-front"},
            section_anchors=anchors,
        )
        self.assertEqual(
            [item.render_intent for item in physical.fragments],
            [
                BoundaryRenderIntent.SOLID,
                BoundaryRenderIntent.OMIT,
                BoundaryRenderIntent.OMIT,
                BoundaryRenderIntent.SOLID,
            ],
        )
        diagrammatic = compute_quadric_boundary_compositing(
            (source,),
            spans,
            paint_policy=QuadricPaintPolicy.DIAGRAMMATIC,
            parent_item_ids=parent,
            parent_relations=parent_relations,
            surface_item_by_id={"solid": "surface-front"},
            section_anchors=anchors,
        )
        self.assertTrue(
            all(
                diagrammatic.draw_order.index(item.item_id)
                > diagrammatic.draw_order.index("outline-front")
                for item in diagrammatic.fragments
                if item.visibility_kind is VisibilityKind.HIDDEN
            )
        )
        depth = compute_quadric_boundary_compositing(
            (source,),
            spans,
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            parent_item_ids=parent,
            parent_relations=parent_relations,
            surface_item_by_id={"solid": "surface-front"},
            section_anchors=anchors,
        )
        hidden = [
            item for item in depth.fragments
            if item.visibility_kind is VisibilityKind.HIDDEN
        ]
        behind, between = hidden
        self.assertLess(
            depth.draw_order.index(behind.item_id),
            depth.draw_order.index("surface-back"),
        )
        self.assertLess(
            depth.draw_order.index(between.item_id),
            depth.draw_order.index("surface-front"),
        )
        self.assertGreater(
            depth.draw_order.index(between.item_id),
            depth.draw_order.index("plane-between"),
        )

    def test_projected_crossing_splits_and_orders_boundary_fragments(self) -> None:
        far_curve = SegmentCurve(
            "a-far", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        )
        near_curve = SegmentCurve(
            "b-near", (0.0, -1.0, 1.0), (0.0, 1.0, 1.0)
        )
        sources = (
            curve_boundary_source(far_curve),
            curve_boundary_source(near_curve),
        )
        spans = {
            item.source_id: (
                QuadricBoundaryVisibilitySpan(
                    item.curve.domain, VisibilityKind.VISIBLE
                ),
            )
            for item in sources
        }
        crossings = compute_projected_curve_crossings(
            (far_curve, near_curve), IDENTITY_VIEW
        )
        self.assertEqual(len(crossings), 1)
        frame = compute_quadric_boundary_compositing(
            sources,
            spans,
            paint_policy="diagrammatic",
            parent_item_ids=(),
            parent_relations=(),
            surface_item_by_id={},
            crossings=crossings,
        )
        by_source = {
            source.source_id: tuple(
                item
                for item in frame.fragments
                if item.source_id == source.source_id
            )
            for source in sources
        }
        self.assertEqual(len(by_source["a-far"]), 2)
        self.assertEqual(len(by_source["b-near"]), 2)
        crossing = crossings[0]
        far_active = tuple(
            item
            for item in by_source[crossing.far_curve_id]
            if item.interval.contains(
                crossing.first_parameter
                if crossing.far_curve_id == crossing.first_curve_id
                else crossing.second_parameter,
                tolerance=1.0e-12,
            )
        )
        near_active = tuple(
            item
            for item in by_source[crossing.near_curve_id]
            if item.interval.contains(
                crossing.first_parameter
                if crossing.near_curve_id == crossing.first_curve_id
                else crossing.second_parameter,
                tolerance=1.0e-12,
            )
        )
        self.assertTrue(far_active and near_active)
        for farther in far_active:
            for nearer in near_active:
                self.assertLess(
                    frame.draw_order.index(farther.item_id),
                    frame.draw_order.index(nearer.item_id),
                )
        rebuilt = compute_quadric_boundary_compositing(
            sources,
            spans,
            paint_policy="diagrammatic",
            parent_item_ids=(),
            parent_relations=(),
            surface_item_by_id={},
            crossings=crossings,
        )
        self.assertEqual(
            canonical_quadric_boundary_compositing_json(frame),
            canonical_quadric_boundary_compositing_json(rebuilt),
        )
        self.assertEqual(frame.draw_order, rebuilt.draw_order)
        self.assertEqual(
            tuple(item.item_id for item in frame.fragments),
            tuple(item.item_id for item in rebuilt.fragments),
        )

    def test_boundary_frame_is_canonical(self) -> None:
        curve = SegmentCurve("curve:a", (-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        source = curve_boundary_source(curve)
        spans = {
            source.source_id: (
                QuadricBoundaryVisibilitySpan(
                    curve.domain, VisibilityKind.VISIBLE
                ),
            )
        }
        frame = compute_quadric_boundary_compositing(
            (source,),
            spans,
            paint_policy="diagrammatic",
            parent_item_ids=("surface:s",),
            parent_relations=(),
            surface_item_by_id={"s": "surface:s"},
        )
        self.assertEqual(
            canonical_quadric_boundary_compositing_json(frame),
            canonical_quadric_boundary_compositing_json(frame),
        )
        self.assertEqual(frame.draw_order[-1], frame.fragments[0].item_id)


if __name__ == "__main__":
    unittest.main()
