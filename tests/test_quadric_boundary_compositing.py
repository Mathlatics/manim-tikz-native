from __future__ import annotations

from dataclasses import replace
import json
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
    QUADRIC_BOUNDARY_COMPOSITING_SCHEMA,
    QuadricBoundaryCompositingError,
    QuadricBoundaryVisibilitySpan,
    canonical_quadric_boundary_compositing_json,
    compute_boundary_visibility,
    compute_quadric_boundary_compositing,
)
from polyhedron_visibility.quadrics.boundary_section import (
    BoundaryPlaneRelation,
    QuadricBoundarySectionSpan,
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
    ProjectedCurveCrossing,
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
    def test_generator_style_identity_is_canonical_and_non_empty(self) -> None:
        spec = GeneratorBoundarySpec(
            "generator",
            "cone",
            0.4,
            style_id="  style:accent  ",
        )
        self.assertEqual(spec.style_id, "style:accent")
        with self.assertRaisesRegex(ValueError, "style_id must be"):
            GeneratorBoundarySpec(
                "invalid",
                "cone",
                0.4,
                style_id="   ",
            )

    def test_hidden_free_curve_behind_between_plane_uses_back_sheet_lower_bound(
        self,
    ) -> None:
        curve = SegmentCurve(
            "hidden-transition-curve",
            (-0.5, 0.0, 0.0),
            (0.5, 0.0, 0.0),
        )
        source = curve_boundary_source(curve)
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
        frame = compute_quadric_boundary_compositing(
            (source,),
            {
                source.source_id: (
                    QuadricBoundaryVisibilitySpan(
                        curve.domain,
                        VisibilityKind.HIDDEN,
                        ("solid",),
                    ),
                )
            },
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            parent_item_ids=parent,
            parent_relations=tuple(
                QuadricPaintRelation(left, right, "parent")
                for left, right in zip(parent, parent[1:])
            ),
            surface_item_by_id={"solid": "surface-front"},
            section_anchors=anchors,
            section_spans_by_source={
                source.source_id: (
                    QuadricBoundarySectionSpan(
                        curve.domain,
                        BoundaryPlaneRelation.BOUNDARY_BEHIND_PLANE,
                        ("between_surface_sheets",),
                    ),
                )
            },
        )
        fragment = frame.fragments[0]
        ranks = {
            item_id: index for index, item_id in enumerate(frame.draw_order)
        }
        self.assertLess(ranks["surface-back"], ranks[fragment.item_id])
        self.assertLess(ranks[fragment.item_id], ranks["plane-between"])
        self.assertLess(ranks[fragment.item_id], ranks["surface-front"])

    def test_visible_free_curve_behind_section_plane_uses_certified_order(
        self,
    ) -> None:
        curve = SegmentCurve(
            "transition-curve", (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)
        )
        source = curve_boundary_source(curve)
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
        frame = compute_quadric_boundary_compositing(
            (source,),
            {
                source.source_id: (
                    QuadricBoundaryVisibilitySpan(
                        curve.domain, VisibilityKind.VISIBLE
                    ),
                )
            },
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            parent_item_ids=parent,
            parent_relations=tuple(
                QuadricPaintRelation(left, right, "parent")
                for left, right in zip(parent, parent[1:])
            ),
            surface_item_by_id={"solid": "surface-front"},
            section_anchors=anchors,
            section_spans_by_source={
                source.source_id: (
                    QuadricBoundarySectionSpan(
                        curve.domain,
                        BoundaryPlaneRelation.BOUNDARY_BEHIND_PLANE,
                        ("in_front_of_surface",),
                    ),
                )
            },
        )
        fragment = frame.fragments[0]
        self.assertLess(
            frame.draw_order.index("surface-front"),
            frame.draw_order.index(fragment.item_id),
        )
        self.assertLess(
            frame.draw_order.index(fragment.item_id),
            frame.draw_order.index("plane-front"),
        )
        self.assertLess(
            frame.draw_order.index(fragment.item_id),
            frame.draw_order.index("outline-front"),
        )

    def test_visible_curve_outside_patch_is_not_tied_to_plane_outline(self) -> None:
        curve = SegmentCurve(
            "outside-patch-curve", (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)
        )
        source = curve_boundary_source(curve)
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
            anchors.plane_behind,
            anchors.outline_behind,
            anchors.surface_back,
            anchors.plane_outside,
            anchors.outline_outside,
            anchors.plane_between,
            anchors.outline_between,
            anchors.surface_front,
            anchors.plane_front,
            anchors.outline_front,
        )
        frame = compute_quadric_boundary_compositing(
            (source,),
            {
                source.source_id: (
                    QuadricBoundaryVisibilitySpan(
                        curve.domain, VisibilityKind.VISIBLE
                    ),
                )
            },
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            parent_item_ids=parent,
            parent_relations=tuple(
                QuadricPaintRelation(left, right, "parent")
                for left, right in zip(parent, parent[1:])
            ),
            surface_item_by_id={"solid": anchors.surface_front},
            section_anchors=anchors,
            section_spans_by_source={
                source.source_id: (
                    QuadricBoundarySectionSpan(
                        curve.domain,
                        BoundaryPlaneRelation.OUTSIDE_PATCH,
                    ),
                )
            },
        )
        fragment = frame.fragments[0]
        relations = {
            (item.far_item_id, item.near_item_id, item.reason)
            for item in frame.order_relations
        }
        self.assertIn(
            (
                anchors.surface_front,
                fragment.item_id,
                "visible_boundary_outside_section_patch",
            ),
            relations,
        )
        self.assertFalse(
            any(
                far == anchors.outline_front and near == fragment.item_id
                for far, near, _reason in relations
            )
        )

    def test_plane_occluded_owner_silhouette_uses_front_plane_roles(self) -> None:
        cone = ConeSpec(
            "cone",
            (0.0, 0.0, -2.0),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 4.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        source = next(
            item
            for item in build_surface_boundary_sources(
                (cone,), VIEW, include_cap_rims=False
            )
            if item.semantic_kind is BoundarySemanticKind.TRUE_SILHOUETTE
        )
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
            anchors.plane_behind,
            anchors.outline_behind,
            anchors.surface_back,
            anchors.plane_outside,
            anchors.outline_outside,
            anchors.plane_between,
            anchors.outline_between,
            anchors.surface_front,
            anchors.plane_front,
            anchors.outline_front,
        )
        frame = compute_quadric_boundary_compositing(
            (source,),
            {
                source.source_id: (
                    QuadricBoundaryVisibilitySpan(
                        source.curve.domain,
                        VisibilityKind.VISIBLE,
                    ),
                )
            },
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            parent_item_ids=parent,
            parent_relations=tuple(
                QuadricPaintRelation(left, right, "parent")
                for left, right in zip(parent, parent[1:])
            ),
            surface_item_by_id={"cone": "surface-front"},
            section_anchors=anchors,
            section_spans_by_source={
                source.source_id: (
                    QuadricBoundarySectionSpan(
                        source.curve.domain,
                        BoundaryPlaneRelation.BOUNDARY_BEHIND_PLANE,
                        ("in_front_of_surface", "outside_projection"),
                    ),
                )
            },
        )
        fragment = frame.fragments[0]
        self.assertIs(
            fragment.surface_visibility_kind,
            VisibilityKind.VISIBLE,
        )
        self.assertIs(
            fragment.effective_visibility_kind,
            VisibilityKind.HIDDEN,
        )
        self.assertTrue(fragment.plane_occluded)
        self.assertEqual(fragment.occluder_surface_ids, ())
        self.assertEqual(
            fragment.render_intent,
            BoundaryRenderIntent.DASHED,
        )
        self.assertLess(
            frame.draw_order.index("surface-front"),
            frame.draw_order.index(fragment.item_id),
        )
        relations = {
            (item.far_item_id, item.near_item_id) for item in frame.order_relations
        }
        self.assertIn((fragment.item_id, "plane-front"), relations)
        self.assertIn((fragment.item_id, "plane-outside"), relations)
        self.assertNotIn((fragment.item_id, "plane-between"), relations)

    def test_section_plane_occlusion_controls_effective_boundary_visibility(
        self,
    ) -> None:
        cone = ConeSpec(
            "plane-occluded-cone",
            (0.0, 0.0, -2.0),
            (0.0, 0.0, 1.0),
            pi / 6.0,
            (0.0, 4.0),
            radial_axis=(1.0, 0.0, 0.0),
        )
        source = next(
            item
            for item in build_surface_boundary_sources(
                (cone,), VIEW, include_cap_rims=False
            )
            if item.semantic_kind is BoundarySemanticKind.TRUE_SILHOUETTE
        )
        self.assertIs(
            source.occlusion_scope,
            BoundaryOcclusionScope.EXTERNAL_ONLY,
        )
        domain = source.curve.domain
        first = domain.start + 0.25 * domain.length
        second = domain.start + 0.50 * domain.length
        section_spans = {
            source.source_id: (
                QuadricBoundarySectionSpan(
                    ParameterInterval(domain.start, first),
                    BoundaryPlaneRelation.OUTSIDE_PATCH,
                ),
                QuadricBoundarySectionSpan(
                    ParameterInterval(first, second),
                    BoundaryPlaneRelation.BOUNDARY_IN_FRONT_OF_PLANE,
                    ("behind_surface", "outside_projection"),
                ),
                QuadricBoundarySectionSpan(
                    ParameterInterval(second, domain.end),
                    BoundaryPlaneRelation.BOUNDARY_BEHIND_PLANE,
                    ("in_front_of_surface", "outside_projection"),
                ),
            )
        }
        spans = {
            source.source_id: (
                QuadricBoundaryVisibilitySpan(
                    domain,
                    VisibilityKind.VISIBLE,
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
        parent = tuple(getattr(anchors, name) for name in anchors.__dataclass_fields__)
        parent_relations = tuple(
            QuadricPaintRelation(left, right, "parent")
            for left, right in zip(parent, parent[1:])
        )

        def build(policy: QuadricPaintPolicy):
            return compute_quadric_boundary_compositing(
                (source,),
                spans,
                paint_policy=policy,
                parent_item_ids=parent,
                parent_relations=parent_relations,
                surface_item_by_id={cone.surface_id: anchors.surface_front},
                section_anchors=anchors,
                section_spans_by_source=section_spans,
            )

        for policy, hidden_intent in (
            (QuadricPaintPolicy.PHYSICAL, BoundaryRenderIntent.OMIT),
            (QuadricPaintPolicy.DIAGRAMMATIC, BoundaryRenderIntent.DASHED),
            (
                QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
                BoundaryRenderIntent.DASHED,
            ),
        ):
            with self.subTest(policy=policy.value):
                frame = build(policy)
                outside, in_front, behind = sorted(
                    frame.fragments,
                    key=lambda item: item.interval.start,
                )
                for visible in (outside, in_front):
                    self.assertIs(
                        visible.surface_visibility_kind,
                        VisibilityKind.VISIBLE,
                    )
                    self.assertIs(
                        visible.effective_visibility_kind,
                        VisibilityKind.VISIBLE,
                    )
                    self.assertFalse(visible.plane_occluded)
                    self.assertEqual(
                        visible.render_intent,
                        BoundaryRenderIntent.SOLID,
                    )
                self.assertIs(
                    behind.surface_visibility_kind,
                    VisibilityKind.VISIBLE,
                )
                self.assertIs(
                    behind.effective_visibility_kind,
                    VisibilityKind.HIDDEN,
                )
                self.assertTrue(behind.plane_occluded)
                self.assertEqual(behind.occluder_surface_ids, ())
                self.assertEqual(
                    behind.plane_occluder_item_ids,
                    (anchors.plane_front, anchors.plane_outside),
                )
                self.assertEqual(behind.render_intent, hidden_intent)

        physical = build(QuadricPaintPolicy.PHYSICAL)
        physical_behind = max(
            physical.fragments,
            key=lambda item: item.interval.start,
        )
        self.assertFalse(physical_behind.painted)

        diagrammatic = build(QuadricPaintPolicy.DIAGRAMMATIC)
        diagrammatic_behind = max(
            diagrammatic.fragments,
            key=lambda item: item.interval.start,
        )
        self.assertGreater(
            diagrammatic.draw_order.index(diagrammatic_behind.item_id),
            diagrammatic.draw_order.index(anchors.outline_front),
        )

        depth_aware = build(QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC)
        depth_behind = max(
            depth_aware.fragments,
            key=lambda item: item.interval.start,
        )
        self.assertLess(
            depth_aware.draw_order.index(anchors.surface_front),
            depth_aware.draw_order.index(depth_behind.item_id),
        )
        for plane_item in depth_behind.plane_occluder_item_ids:
            self.assertLess(
                depth_aware.draw_order.index(depth_behind.item_id),
                depth_aware.draw_order.index(plane_item),
            )
        payload = json.loads(
            canonical_quadric_boundary_compositing_json(depth_aware)
        )
        self.assertEqual(
            payload["schema"],
            "manim-quadric-boundary-compositing/v2",
        )
        self.assertEqual(
            payload["schema"],
            QUADRIC_BOUNDARY_COMPOSITING_SCHEMA,
        )
        fragment_payload = next(
            item
            for item in payload["fragments"]
            if item["itemId"] == depth_behind.item_id
        )
        self.assertNotIn("visibilityKind", fragment_payload)
        self.assertEqual(
            fragment_payload["surfaceVisibilityKind"],
            "visible",
        )
        self.assertEqual(
            fragment_payload["effectiveVisibilityKind"],
            "hidden",
        )
        self.assertTrue(fragment_payload["planeOccluded"])
        self.assertEqual(
            fragment_payload["planeOccluderItemIds"],
            list(depth_behind.plane_occluder_item_ids),
        )

    def test_diagrammatic_hidden_dash_overrides_crossing_depth_order(self) -> None:
        hidden_curve = SegmentCurve(
            "a-hidden",
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        )
        hidden_source = curve_boundary_source(hidden_curve)
        plane = SectionPlane(
            "crossing-plane",
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            u_axis=(1.0, 0.0, 0.0),
        )
        patch = PlaneDisplayPatchSpec(
            "crossing-patch",
            plane.plane_id,
            2.0,
            1.5,
        )
        plane_source = plane_outline_sources(plane, patch)[0]
        self.assertLess(hidden_source.source_id, plane_source.source_id)
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
        parent = tuple(
            getattr(anchors, name) for name in anchors.__dataclass_fields__
        )
        frame = compute_quadric_boundary_compositing(
            (hidden_source, plane_source),
            {
                hidden_source.source_id: (
                    QuadricBoundaryVisibilitySpan(
                        hidden_curve.domain,
                        VisibilityKind.VISIBLE,
                    ),
                ),
                plane_source.source_id: (
                    QuadricBoundaryVisibilitySpan(
                        plane_source.curve.domain,
                        VisibilityKind.VISIBLE,
                        depth_role="in_front_of_surface",
                    ),
                ),
            },
            paint_policy=QuadricPaintPolicy.DIAGRAMMATIC,
            parent_item_ids=parent,
            parent_relations=tuple(
                QuadricPaintRelation(left, right, "parent")
                for left, right in zip(parent, parent[1:])
            ),
            surface_item_by_id={"surface": anchors.surface_front},
            crossings=(
                ProjectedCurveCrossing(
                    "crossing:a-hidden:plane-edge:0",
                    hidden_source.source_id,
                    plane_source.source_id,
                    hidden_curve.domain.midpoint,
                    plane_source.curve.domain.midpoint,
                    (0.0, 0.0),
                    0.0,
                    1.0,
                    hidden_source.source_id,
                    plane_source.source_id,
                ),
            ),
            section_anchors=anchors,
            section_spans_by_source={
                hidden_source.source_id: (
                    QuadricBoundarySectionSpan(
                        hidden_curve.domain,
                        BoundaryPlaneRelation.BOUNDARY_BEHIND_PLANE,
                        ("in_front_of_surface",),
                    ),
                )
            },
        )
        hidden_fragments = tuple(
            item
            for item in frame.fragments
            if item.source_id == hidden_source.source_id and item.painted
        )
        plane_fragments = tuple(
            item
            for item in frame.fragments
            if item.source_id == plane_source.source_id and item.painted
        )
        self.assertTrue(hidden_fragments and plane_fragments)
        self.assertTrue(
            all(
                item.effective_visibility_kind is VisibilityKind.HIDDEN
                and item.render_intent is BoundaryRenderIntent.DASHED
                for item in hidden_fragments
            )
        )
        for plane_fragment in plane_fragments:
            for hidden_fragment in hidden_fragments:
                self.assertLess(
                    frame.draw_order.index(plane_fragment.item_id),
                    frame.draw_order.index(hidden_fragment.item_id),
                )

    def test_surface_item_mapping_must_reference_parent_items(self) -> None:
        curve = SegmentCurve(
            "visible-boundary", (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)
        )
        source = curve_boundary_source(curve)
        with self.assertRaisesRegex(
            QuadricBoundaryCompositingError,
            "references non-parent items",
        ):
            compute_quadric_boundary_compositing(
                (source,),
                {
                    source.source_id: (
                        QuadricBoundaryVisibilitySpan(
                            curve.domain, VisibilityKind.VISIBLE
                        ),
                    )
                },
                paint_policy=QuadricPaintPolicy.PHYSICAL,
                parent_item_ids=("surface:actual",),
                parent_relations=(),
                surface_item_by_id={"actual": "surface:missing"},
            )

    def test_depth_aware_hidden_boundary_is_globally_bracketed(self) -> None:
        curve = SegmentCurve(
            "hidden-boundary", (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)
        )
        source = curve_boundary_source(curve)
        frame = compute_quadric_boundary_compositing(
            (source,),
            {
                source.source_id: (
                    QuadricBoundaryVisibilitySpan(
                        curve.domain,
                        VisibilityKind.HIDDEN,
                        ("near",),
                    ),
                )
            },
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            parent_item_ids=("surface:far", "surface:near"),
            parent_relations=(
                QuadricPaintRelation(
                    "surface:far", "surface:near", "certified_surface_order"
                ),
            ),
            surface_item_by_id={
                "far": "surface:far",
                "near": "surface:near",
            },
        )
        fragment = frame.fragments[0]
        ranks = {
            item_id: index for index, item_id in enumerate(frame.draw_order)
        }
        self.assertLess(ranks["surface:far"], ranks[fragment.item_id])
        self.assertLess(ranks[fragment.item_id], ranks["surface:near"])
        self.assertIn(
            (
                "surface:far",
                fragment.item_id,
                "depth_aware_hidden_boundary_after_farther_surface",
            ),
            {
                (item.far_item_id, item.near_item_id, item.reason)
                for item in frame.order_relations
            },
        )

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
                if item.effective_visibility_kind is VisibilityKind.HIDDEN
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
            if item.effective_visibility_kind is VisibilityKind.HIDDEN
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
        with self.assertRaisesRegex(
            QuadricBoundaryCompositingError,
            "invalid quadric boundary compositing schema",
        ):
            replace(
                frame,
                schema="manim-quadric-boundary-compositing/v1",
            )


if __name__ == "__main__":
    unittest.main()
