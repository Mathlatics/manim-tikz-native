from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from math import atan, cos, isfinite, sin, sqrt, tau
import unittest

import numpy as np

from polyhedron_visibility.geometry import GeometryContext, GeometryQuantity
from polyhedron_visibility.parallel_solver import ParallelView
from polyhedron_visibility.topology import ParameterInterval
from polyhedron_visibility.visibility import VisibilityKind
from polyhedron_visibility.quadrics.boundary_compositing import (
    BoundaryOcclusionScope,
    BoundarySemanticKind,
    BoundarySourceKind,
    QuadricBoundaryCompositingError,
    compute_boundary_visibility,
    compute_quadric_boundary_crossings,
)
from polyhedron_visibility.quadrics.compositing import QuadricPaintPolicy
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    CylinderSpec,
    PlaneDisplayPatchSpec,
    SectionPlane,
    SphereSpec,
)
from polyhedron_visibility.quadrics.critical import (
    CriticalEventKind,
    compute_curve_critical_events,
)
from polyhedron_visibility.quadrics.curves import EllipseArcCurve, SegmentCurve
from polyhedron_visibility.quadrics.global_occlusion import (
    compute_global_quadric_frame,
)
from polyhedron_visibility.quadrics.nested_tangent_compositing import (
    NestedTangentCompositingError,
    NestedTangentSphereSpec,
    _certify_contact,
    compute_nested_tangent_parent_frame,
)
from polyhedron_visibility.quadrics.scene_occlusion import (
    SceneOcclusionError,
    SceneOcclusionPath,
    SceneOcclusionRequest,
    SceneSectionSpec,
    _nested_silhouette_support_tangencies,
    compute_scene_occlusion_frame,
)
from polyhedron_visibility.quadrics.section_compositing import (
    PlaneDepthRole,
    PlanePatchProjectionKind,
    compute_quadric_section_compositing,
)
from polyhedron_visibility.quadrics.sections import (
    compute_quadric_section_boundary_curves,
)
from polyhedron_visibility.quadrics.surface_boundaries import (
    build_surface_boundary_sources,
    curve_boundary_source,
    plane_outline_sources,
    section_curve_boundary_source,
)


_RADIUS = 1.45
_AXIAL_RANGE = (-3.0, 5.8)
_CONE_SLOPE = _RADIUS / abs(_AXIAL_RANGE[0])
_PLANE_NORMAL = (0.25, 0.0, sqrt(1.0 - 0.25 * 0.25))
_PLANE_OFFSET = 0.20
_PLANE = SectionPlane(
    "scene-plane",
    tuple(_PLANE_OFFSET * value for value in _PLANE_NORMAL),
    _PLANE_NORMAL,
    (0.0, 1.0, 0.0),
)
_PATCH = PlaneDisplayPatchSpec("scene-patch", _PLANE.plane_id, 3.35, 3.55)
_DEPTH = np.asarray((0.75, -1.25, 0.55), dtype=float)
_DEPTH /= np.linalg.norm(_DEPTH)
_RIGHT = np.asarray((-_DEPTH[1], _DEPTH[0], 0.0), dtype=float)
_RIGHT /= np.linalg.norm(_RIGHT)
_UP = np.cross(_DEPTH, _RIGHT)
_UP /= np.linalg.norm(_UP)
_VIEW = ParallelView.from_matrix(np.vstack((0.65 * _RIGHT, 0.65 * _UP, _DEPTH)))


def _geometry(progress: float):
    slope = _CONE_SLOPE * (1.0 - progress)
    normalization = sqrt(1.0 + slope * slope)
    intercept = _RADIUS / normalization
    gradient = slope / normalization
    records = []
    for side in (-1, 1):
        denominator = _PLANE_NORMAL[2] - side * gradient
        center_z = (_PLANE_OFFSET + side * intercept) / denominator
        radius = intercept + gradient * center_z
        contact_radius = (_RADIUS + slope * center_z) / (1.0 + slope * slope)
        contact_z = (center_z - slope * _RADIUS) / (1.0 + slope * slope)
        records.append((side, center_z, radius, contact_radius, contact_z))
    if slope == 0.0:
        mother = CylinderSpec(
            "mother",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            _RADIUS,
            _AXIAL_RANGE,
            radial_axis=(1.0, 0.0, 0.0),
        )
    else:
        apex_z = -_RADIUS / slope
        mother = ConeSpec(
            "mother",
            (0.0, 0.0, apex_z),
            (0.0, 0.0, 1.0),
            atan(slope),
            (_AXIAL_RANGE[0] - apex_z, _AXIAL_RANGE[1] - apex_z),
            radial_axis=(1.0, 0.0, 0.0),
            model=ConeModel.OPEN_SINGLE,
        )
    return mother, tuple(records)


def _request(progress: float) -> SceneOcclusionRequest:
    mother, records = _geometry(progress)
    spheres = []
    sources = []
    bindings = []
    for side, center_z, radius, contact_radius, contact_z in records:
        sphere_id = f"sphere:{side:+d}"
        contact_id = f"contact:{side:+d}"
        sphere = SphereSpec(sphere_id, (0.0, 0.0, center_z), radius)
        contact = EllipseArcCurve(
            contact_id,
            (0.0, 0.0, contact_z),
            (contact_radius, 0.0, 0.0),
            (0.0, contact_radius, 0.0),
        )
        spheres.append(sphere)
        sources.append(
            curve_boundary_source(
                contact,
                source_kind=BoundarySourceKind.ANALYTIC_CURVE,
                semantic_kind=BoundarySemanticKind.TEACHING_FEATURE,
                occlusion_scope=BoundaryOcclusionScope.OWNER_AND_EXTERNAL,
                owner_id=sphere_id,
                owner_surface_id=sphere_id,
                style_id="style:contact",
            )
        )
        bindings.append(
            NestedTangentSphereSpec(
                sphere_id,
                mother.surface_id,
                contact_id,
                f"item:{sphere_id}",
            )
        )
    section_curves = compute_quadric_section_boundary_curves(
        "scene-section",
        mother,
        _PLANE,
    )
    sources.extend(
        section_curve_boundary_source(
            curve,
            mother,
            _PLANE,
            section_id="scene-section",
            authoritative_curves=section_curves,
            style_id="style:section",
        )
        for curve in section_curves
    )
    return SceneOcclusionRequest(
        "nested-scene",
        (mother, *spheres),
        _VIEW,
        tuple(sources),
        SceneSectionSpec(mother.surface_id, _PLANE, _PATCH),
        tuple(bindings),
        QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
        max_chord_error=0.14,
        max_surface_segments=2048,
    )


def _request_with_intrinsic_silhouettes(
    progress: float,
) -> tuple[SceneOcclusionRequest, tuple]:
    request = _request(progress)
    silhouettes = tuple(
        replace(source, style_id="style:test-nested-silhouette")
        for source in build_surface_boundary_sources(
            request.surfaces,
            request.view,
            include_cap_rims=False,
            include_silhouettes=True,
        )
    )
    return (
        replace(
            request,
            boundary_sources=(*request.boundary_sources, *silhouettes),
        ),
        silhouettes,
    )


@lru_cache(maxsize=8)
def _frame(progress: float):
    return compute_scene_occlusion_frame(_request(progress))


class QuadricSceneOcclusionTests(unittest.TestCase):
    def test_canonical_nested_silhouettes_emit_geometric_critical_evidence(
        self,
    ) -> None:
        # This is close enough to the cylinder endpoint that the generic
        # tan-half crossing polynomial used to split one exact tangency into
        # two roots on macOS as well as Linux.
        request, silhouettes = _request_with_intrinsic_silhouettes(0.99999999)
        frame = compute_scene_occlusion_frame(request)
        context = frame.global_frame.geometry_context
        certificates = _nested_silhouette_support_tangencies(
            request.boundary_sources,
            request.surfaces,
            frame.nested_parent_frame,
            request.view,
            context=context,
        )
        sphere_sources = tuple(
            source
            for source in silhouettes
            if source.owner_surface_id.startswith("sphere:")
        )
        self.assertEqual(
            set(certificates),
            {source.source_id for source in sphere_sources},
        )
        boundary_epsilon = context.epsilon(
            GeometryQuantity.BOUNDARY
        )
        parameter_epsilon = context.epsilon(
            GeometryQuantity.PARAMETER
        )
        source_by_id = {
            source.source_id: source for source in request.boundary_sources
        }
        mother_id = frame.nested_parent_frame.mother_surface_id
        certified_pairs: set[tuple[str, str]] = set()
        all_certificates = tuple(
            item
            for source_id in sorted(certificates)
            for item in certificates[source_id]
        )
        mother_source_ids = {
            source.source_id
            for source in silhouettes
            if source.owner_surface_id == mother_id
        }
        self.assertEqual(
            {
                (item.curve_id, item.witness_curve_id)
                for item in all_certificates
            },
            {
                (sphere_source.source_id, mother_source_id)
                for sphere_source in sphere_sources
                for mother_source_id in mother_source_ids
            },
        )
        self.assertEqual(
            len({item.crossing_id for item in all_certificates}),
            4,
        )
        for source in sphere_sources:
            with self.subTest(source_id=source.source_id):
                items = certificates[source.source_id]
                self.assertEqual(len(items), 2)
                self.assertEqual(
                    len({item.witness_curve_id for item in items}),
                    2,
                )
                self.assertTrue(
                    all(item.world_residual <= boundary_epsilon for item in items)
                )
                for item in items:
                    self.assertEqual(
                        item.contact_curve_id,
                        f"contact:{source.owner_surface_id.removeprefix('sphere:')}",
                    )
                    contact_curve = source_by_id[item.contact_curve_id].curve
                    self.assertTrue(
                        contact_curve.domain.contains(
                            item.contact_parameter,
                            tolerance=parameter_epsilon,
                        )
                    )
                    self.assertLessEqual(
                        float(
                            np.linalg.norm(
                                np.asarray(
                                    contact_curve.point(item.contact_parameter)
                                )
                                - np.asarray(item.world_point)
                            )
                        ),
                        boundary_epsilon,
                    )
                    self.assertLessEqual(
                        float(
                            np.linalg.norm(
                                np.asarray(source.curve.point(item.parameter))
                                - np.asarray(item.world_point)
                            )
                        ),
                        boundary_epsilon,
                    )
                    witness_curve = source_by_id[item.witness_curve_id].curve
                    self.assertLessEqual(
                        float(
                            np.linalg.norm(
                                np.asarray(
                                    witness_curve.point(item.witness_parameter)
                                )
                                - np.asarray(item.world_point)
                            )
                        ),
                        boundary_epsilon,
                    )
                    triple = sorted(
                        {
                            item.curve_id,
                            item.witness_curve_id,
                            item.contact_curve_id,
                        }
                    )
                    certified_pairs.update(
                        {
                            (triple[0], triple[1]),
                            (triple[0], triple[2]),
                            (triple[1], triple[2]),
                        }
                    )
                self.assertGreater(
                    float(
                        np.linalg.norm(
                            np.asarray(items[1].world_point)
                            - np.asarray(items[0].world_point)
                        )
                    ),
                    boundary_epsilon,
                )

                selected_surfaces = tuple(
                    surface
                    for surface in request.surfaces
                    if surface.surface_id != source.owner_surface_id
                )
                events = compute_curve_critical_events(
                    source.curve,
                    selected_surfaces,
                    request.view,
                    context=context,
                    _nested_silhouette_tangencies=items,
                )
                evidence = tuple(
                    item
                    for event in events
                    for item in event.evidence
                    if item.surface_id == mother_id
                )
                support = tuple(
                    item
                    for item in evidence
                    if item.kind is CriticalEventKind.SUPPORT_TANGENCY
                    and item.equation.startswith(
                        "nested_tangent_silhouette_support:"
                    )
                )
                contact = tuple(
                    item
                    for item in evidence
                    if item.kind
                    is CriticalEventKind.CURVE_SURFACE_INTERSECTION
                    and item.equation.startswith(
                        "nested_tangent_silhouette_contact:"
                    )
                )
                self.assertEqual(len(support), 2)
                self.assertEqual(len(contact), 2)
                for item in (*support, *contact):
                    self.assertEqual(item.chart, "geometric_certificate")
                    self.assertEqual(item.coefficients, ())
                    self.assertLessEqual(item.residual, boundary_epsilon)
                self.assertNotIn(
                    "ray_discriminant",
                    {item.equation for item in evidence},
                )
                self.assertNotIn(
                    "curve_on_surface",
                    {item.equation for item in evidence},
                )
                self.assertIn(
                    "ray_linear_coefficient",
                    {item.equation for item in evidence},
                )

                self.assertTrue(
                    all(
                        fragment.interval.length > 100.0 * parameter_epsilon
                        for fragment in frame.boundary_frame.fragments
                        if fragment.source_id == source.source_id
                    )
                )

        crossing_pairs = {
            (item.first_curve_id, item.second_curve_id)
            for item in frame.boundary_frame.crossings
        }
        self.assertTrue(certified_pairs.isdisjoint(crossing_pairs))

    def test_nested_nonordering_pair_map_rejects_incomplete_lineage(
        self,
    ) -> None:
        request, _silhouettes = _request_with_intrinsic_silhouettes(0.9999)
        frame = compute_scene_occlusion_frame(request)
        certificates = _nested_silhouette_support_tangencies(
            request.boundary_sources,
            request.surfaces,
            frame.nested_parent_frame,
            request.view,
            context=frame.global_frame.geometry_context,
        )
        source_id = sorted(certificates)[0]
        items = certificates[source_id]
        forged = {
            **certificates,
            source_id: (
                items[0],
                replace(
                    items[1],
                    witness_curve_id=items[0].witness_curve_id,
                ),
            ),
        }
        spans = compute_boundary_visibility(
            request.boundary_sources,
            request.surfaces,
            request.view,
            context=frame.global_frame.geometry_context,
            _nested_silhouette_tangencies_by_source=certificates,
        )

        with self.assertRaisesRegex(
            QuadricBoundaryCompositingError,
            "requires two certified support tangencies",
        ):
            compute_quadric_boundary_crossings(
                request.boundary_sources,
                spans,
                request.view,
                paint_policy=request.paint_policy,
                context=frame.global_frame.geometry_context,
                _nested_silhouette_tangencies_by_source=forged,
            )

    def test_nested_silhouette_certificate_rechecks_its_contact_circle(
        self,
    ) -> None:
        request, _silhouettes = _request_with_intrinsic_silhouettes(0.9999)
        frame = compute_scene_occlusion_frame(request)
        target = next(
            source
            for source in request.boundary_sources
            if source.source_id == "contact:+1"
        )
        curve = target.curve
        self.assertIsInstance(curve, EllipseArcCurve)
        shifted = EllipseArcCurve(
            curve.curve_id,
            (
                curve.center[0] + 1.0e-4,
                curve.center[1],
                curve.center[2],
            ),
            curve.first_axis,
            curve.second_axis,
            domain=curve.domain,
        )

        with self.assertRaisesRegex(
            SceneOcclusionError,
            "does not reconstruct the nested tangency witness",
        ):
            _nested_silhouette_support_tangencies(
                tuple(
                    replace(target, curve=shifted)
                    if source.source_id == target.source_id
                    else source
                    for source in request.boundary_sources
                ),
                request.surfaces,
                frame.nested_parent_frame,
                request.view,
                context=frame.global_frame.geometry_context,
            )

    def test_tampered_canonical_nested_silhouette_fails_closed(self) -> None:
        request, silhouettes = _request_with_intrinsic_silhouettes(0.5)
        target = next(
            source
            for source in silhouettes
            if source.owner_surface_id == "sphere:+1"
        )
        shifted_curve = EllipseArcCurve(
            target.curve.curve_id,
            (
                target.curve.center[0] + 0.01,
                target.curve.center[1],
                target.curve.center[2],
            ),
            target.curve.first_axis,
            target.curve.second_axis,
            domain=target.curve.domain,
        )
        variants = {
            "curve": replace(target, curve=shifted_curve),
            "owner": replace(target, owner_id="forged-sphere-owner"),
            "surface": replace(target, owner_surface_id="mother"),
            "kind": replace(
                target,
                source_kind=BoundarySourceKind.ANALYTIC_CURVE,
            ),
            "semantic": replace(
                target,
                semantic_kind=BoundarySemanticKind.SURFACE_BOUNDARY,
            ),
            "scope": replace(
                target,
                occlusion_scope=BoundaryOcclusionScope.OWNER_AND_EXTERNAL,
            ),
        }
        for label, forged in variants.items():
            with self.subTest(label=label), self.assertRaisesRegex(
                SceneOcclusionError,
                "does not match its canonical surface boundary",
            ):
                compute_scene_occlusion_frame(
                    replace(
                        request,
                        boundary_sources=tuple(
                            forged if source.source_id == target.source_id else source
                            for source in request.boundary_sources
                        ),
                    )
                )

    def test_partial_nested_silhouette_catalog_uses_generic_solver(self) -> None:
        request, silhouettes = _request_with_intrinsic_silhouettes(0.5)
        omitted_id = "boundary:mother:silhouette:generator:1"
        partial_sources = tuple(
            source
            for source in request.boundary_sources
            if source.source_id != omitted_id
        )
        partial_request = replace(request, boundary_sources=partial_sources)
        frame = compute_scene_occlusion_frame(partial_request)
        self.assertIs(
            frame.dispatch_path,
            SceneOcclusionPath.NESTED_TANGENT_SECTION,
        )
        self.assertIn(
            "boundary:sphere:+1:silhouette",
            {source.source_id for source in frame.boundary_frame.sources},
        )
        self.assertEqual(
            _nested_silhouette_support_tangencies(
                partial_request.boundary_sources,
                partial_request.surfaces,
                frame.nested_parent_frame,
                partial_request.view,
                context=frame.global_frame.geometry_context,
            ),
            {},
        )

    def test_nested_plane_outline_requires_the_complete_finite_patch(self) -> None:
        request = _request(0.5)
        outlines = plane_outline_sources(
            _PLANE,
            _PATCH,
            occlusion_scope=BoundaryOcclusionScope.ALL_SURFACES,
        )

        with self.assertRaisesRegex(
            SceneOcclusionError,
            "cover the finite patch exactly",
        ):
            compute_scene_occlusion_frame(
                replace(
                    request,
                    boundary_sources=(*request.boundary_sources, *outlines[:3]),
                )
            )

    def test_nested_plane_outline_combines_mother_roles_and_sphere_occlusion(
        self,
    ) -> None:
        request = _request(0.5)
        outlines = plane_outline_sources(
            _PLANE,
            _PATCH,
            occlusion_scope=BoundaryOcclusionScope.ALL_SURFACES,
        )
        frame = compute_scene_occlusion_frame(
            replace(
                request,
                boundary_sources=(*request.boundary_sources, *outlines),
            )
        )
        boundary = frame.boundary_frame
        rank = {item_id: index for index, item_id in enumerate(frame.draw_order)}
        mother_id = frame.section_frame.surface_id
        items = frame.section_frame.paint_items
        fill_by_role = {
            PlaneDepthRole.BEHIND_SURFACE: items.plane_behind,
            PlaneDepthRole.OUTSIDE_PROJECTION: items.plane_outside,
            PlaneDepthRole.BETWEEN_SURFACE_SHEETS: items.plane_between,
            PlaneDepthRole.IN_FRONT_OF_SURFACE: items.plane_front,
        }
        outline_by_role = items.outline_by_role
        sphere_items = frame.nested_parent_frame.surface_items
        fragments = tuple(
            item
            for item in boundary.fragments
            if item.source_id in {source.source_id for source in outlines}
        )
        self.assertEqual(len(fragments), 14)
        combined_occlusion = False
        hidden_roles = {
            PlaneDepthRole.BEHIND_SURFACE,
            PlaneDepthRole.BETWEEN_SURFACE_SHEETS,
        }
        by_source = {
            source.source_id: tuple(
                sorted(
                    (
                        item
                        for item in fragments
                        if item.source_id == source.source_id
                    ),
                    key=lambda item: item.interval.start,
                )
            )
            for source in outlines
        }
        for source in outlines:
            pieces = by_source[source.source_id]
            self.assertTrue(pieces)
            self.assertAlmostEqual(
                pieces[0].interval.start,
                source.curve.domain.start,
            )
            self.assertAlmostEqual(
                pieces[-1].interval.end,
                source.curve.domain.end,
            )
            for first, second in zip(pieces, pieces[1:]):
                self.assertAlmostEqual(first.interval.end, second.interval.start)

            for fragment in pieces:
                role = PlaneDepthRole(fragment.depth_role)
                if role in hidden_roles:
                    self.assertIn(mother_id, fragment.occluder_surface_ids)
                else:
                    self.assertNotIn(mother_id, fragment.occluder_surface_ids)
                if mother_id in fragment.occluder_surface_ids and any(
                    item.startswith("sphere:")
                    for item in fragment.occluder_surface_ids
                ):
                    combined_occlusion = True
                if fragment.painted:
                    self.assertLess(
                        rank[fill_by_role[role]],
                        rank[fragment.item_id],
                    )
                    self.assertLess(
                        rank[fragment.item_id],
                        rank[outline_by_role[role]],
                    )
                for occluder_id in fragment.occluder_surface_ids:
                    if occluder_id.startswith("sphere:"):
                        self.assertLess(
                            rank[fragment.item_id],
                            rank[sphere_items[occluder_id]],
                        )
        self.assertTrue(combined_occlusion)

    def test_single_section_is_the_unchanged_existing_fast_path(self) -> None:
        mother, _records = _geometry(0.5)
        request = SceneOcclusionRequest(
            "single-scene",
            (mother,),
            _VIEW,
            section=SceneSectionSpec(mother.surface_id, _PLANE, _PATCH),
            paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            max_chord_error=0.14,
            max_surface_segments=2048,
        )
        coordinated = compute_scene_occlusion_frame(request)
        direct_global = compute_global_quadric_frame(
            (),
            (mother,),
            _VIEW,
            paint_policy=request.paint_policy,
            max_chord_error=request.max_chord_error,
            max_segments=request.max_surface_segments,
        )
        direct_section = compute_quadric_section_compositing(
            direct_global.frame,
            mother,
            _PLANE,
            _PATCH,
            _VIEW,
            context=direct_global.geometry_context,
            max_screen_error=request.max_chord_error,
        )

        self.assertIs(coordinated.dispatch_path, SceneOcclusionPath.SINGLE_SECTION)
        self.assertEqual(coordinated.global_frame, direct_global)
        self.assertEqual(coordinated.section_frame, direct_section)
        self.assertEqual(coordinated.draw_order, direct_section.draw_order)
        self.assertIsNone(coordinated.boundary_frame)
        self.assertTrue(coordinated.physical_surface_visibility_authoritative)

    def test_cone_to_cylinder_uses_one_nested_dispatch_and_exact_curves(self) -> None:
        for progress in (0.0, 0.5, 0.9999, 1.0 - 1.0e-12, 1.0):
            with self.subTest(progress=progress):
                frame = _frame(progress)
                self.assertIs(
                    frame.dispatch_path,
                    SceneOcclusionPath.NESTED_TANGENT_SECTION,
                )
                self.assertTrue(frame.surface_layering_authoritative)
                self.assertTrue(frame.curve_visibility_authoritative)
                self.assertFalse(frame.physical_surface_visibility_authoritative)
                self.assertIsNotNone(frame.boundary_frame)
                self.assertEqual(len(frame.nested_parent_frame.contacts), 2)
                self.assertEqual(
                    len(
                        frame.nested_parent_frame.sphere_pair_frame
                        .separation_evidence
                    ),
                    1,
                )
                for contact in frame.nested_parent_frame.contacts:
                    self.assertLess(contact.max_sphere_contact_residual, 1.0e-10)
                    self.assertLess(contact.max_mother_contact_residual, 1.0e-10)
                    self.assertLess(contact.max_normal_cross_residual, 1.0e-10)
                    self.assertTrue(isfinite(contact.plane_ray_parameter))

                source_ids = {
                    item.source_id for item in frame.boundary_frame.sources
                }
                self.assertEqual(
                    {"contact:-1", "contact:+1"}.intersection(source_ids),
                    {"contact:-1", "contact:+1"},
                )
                self.assertTrue(
                    any(
                        source_id.startswith("scene-section:")
                        for source_id in source_ids
                    )
                )

    def test_contact_circle_is_independent_and_straddles_surfaces(self) -> None:
        frame = _frame(1.0)
        boundary = frame.boundary_frame
        rank = {item_id: index for index, item_id in enumerate(frame.draw_order)}
        parent = frame.nested_parent_frame
        mother_back = frame.section_frame.paint_items.surface_back
        mother_front = frame.section_frame.paint_items.surface_front
        source_map = {item.source_id: item for item in boundary.sources}

        for contact in parent.contacts:
            fragments = tuple(
                item
                for item in boundary.fragments
                if item.source_id == contact.contact_source_id and item.painted
            )
            hidden = tuple(
                item
                for item in fragments
                if item.surface_visibility_kind is VisibilityKind.HIDDEN
            )
            visible = tuple(
                item
                for item in fragments
                if item.surface_visibility_kind is VisibilityKind.VISIBLE
            )
            self.assertEqual(
                source_map[contact.contact_source_id].owner_surface_id,
                contact.sphere_surface_id,
            )
            self.assertTrue(hidden)
            self.assertTrue(visible)
            self.assertNotIn(contact.contact_source_id, parent.parent_item_ids)
            for fragment in hidden:
                self.assertLess(rank[mother_back], rank[fragment.item_id])
                self.assertLess(rank[fragment.item_id], rank[contact.sphere_item_id])
                self.assertLess(rank[fragment.item_id], rank[mother_front])
            for fragment in visible:
                self.assertLess(rank[contact.sphere_item_id], rank[fragment.item_id])
                self.assertLess(rank[mother_front], rank[fragment.item_id])

    def test_shifted_full_circle_domain_is_valid_contact_evidence(self) -> None:
        request = _request(0.5)
        original = next(
            item
            for item in request.boundary_sources
            if item.source_id == "contact:+1"
        )
        shifted_curve = replace(
            original.curve,
            domain=ParameterInterval(1.0, 1.0 + tau),
        )
        shifted_source = replace(original, curve=shifted_curve)
        sources = tuple(
            shifted_source if item.source_id == original.source_id else item
            for item in request.boundary_sources
        )

        frame = compute_scene_occlusion_frame(
            SceneOcclusionRequest(
                request.scene_id,
                request.surfaces,
                request.view,
                sources,
                request.section,
                request.tangent_spheres,
                request.paint_policy,
                max_chord_error=request.max_chord_error,
                max_surface_segments=request.max_surface_segments,
            )
        )

        self.assertEqual(
            {item.contact_source_id for item in frame.nested_parent_frame.contacts},
            {"contact:-1", "contact:+1"},
        )

    def test_free_curve_does_not_cycle_with_disjoint_outside_plane_role(self) -> None:
        request = _request(0.5)
        cases = (
            (
                "free-cycle:outside-role",
                (-0.3072790817028502, -1.92455735302171, -0.39160673),
                (-0.3004191382971498, -1.92044138697829, -0.39160673),
                "boundary_behind_plane",
            ),
            (
                "free-cycle:outside-patch",
                (-0.4444242198029296, 0.04223849476221178, -1.7437402990762458),
                (-0.4459747740666637, -0.006372052461407044, -1.7553424747384025),
                "outside_patch",
            ),
            (
                "free-cycle:behind-plane",
                (-0.991316752835783, 1.6742756420870277, 0.39450090388630404),
                (-0.9601660589676434, 1.7056170858994237, 0.3711054305786771),
                "boundary_behind_plane",
            ),
        )
        for curve_id, start, end, expected_plane_relation in cases:
            with self.subTest(curve_id=curve_id):
                curve = SegmentCurve(curve_id, start, end)
                source = curve_boundary_source(curve, style_id="style:free")

                frame = compute_scene_occlusion_frame(
                    SceneOcclusionRequest(
                        request.scene_id,
                        request.surfaces,
                        request.view,
                        (*request.boundary_sources, source),
                        request.section,
                        request.tangent_spheres,
                        request.paint_policy,
                        max_chord_error=request.max_chord_error,
                        max_surface_segments=request.max_surface_segments,
                    )
                )

                self.assertIn(
                    source.source_id,
                    {item.source_id for item in frame.boundary_frame.sources},
                )
                fragment = next(
                    item
                    for item in frame.boundary_frame.painted_fragments
                    if item.source_id == source.source_id
                )
                self.assertEqual(
                    fragment.plane_relation,
                    expected_plane_relation,
                )
                rank = {
                    item_id: index
                    for index, item_id in enumerate(frame.draw_order)
                }
                if fragment.plane_relation == "outside_patch":
                    self.assertLess(
                        rank[frame.section_frame.paint_items.surface_back],
                        rank[fragment.item_id],
                    )
                    self.assertLess(
                        rank[fragment.item_id],
                        rank[frame.section_frame.paint_items.surface_front],
                    )
                elif fragment.plane_depth_roles == ("behind_surface",):
                    self.assertLess(
                        rank[fragment.item_id],
                        rank[frame.section_frame.paint_items.plane_behind],
                    )
                    self.assertLess(
                        rank[frame.section_frame.paint_items.plane_behind],
                        rank[frame.section_frame.paint_items.surface_back],
                    )

    def test_section_fragments_name_and_obey_sphere_occluders(self) -> None:
        frame = _frame(0.5)
        rank = {item_id: index for index, item_id in enumerate(frame.draw_order)}
        sphere_items = frame.nested_parent_frame.surface_items
        section_fragments = tuple(
            item
            for item in frame.boundary_frame.fragments
            if item.source_id.startswith("scene-section:")
        )
        sphere_hidden = tuple(
            item
            for item in section_fragments
            if any(owner.startswith("sphere:") for owner in item.occluder_surface_ids)
        )
        self.assertTrue(sphere_hidden)
        for fragment in sphere_hidden:
            for surface_id in fragment.occluder_surface_ids:
                if surface_id.startswith("sphere:"):
                    self.assertLess(
                        rank[fragment.item_id],
                        rank[sphere_items[surface_id]],
                    )

    def test_unregistered_multi_surface_section_fails_closed(self) -> None:
        nested = _request(0.5)
        with self.assertRaisesRegex(
            SceneOcclusionError,
            "require exactly two registered tangent spheres",
        ):
            compute_scene_occlusion_frame(
                SceneOcclusionRequest(
                    "unregistered",
                    nested.surfaces,
                    nested.view,
                    section=nested.section,
                    max_chord_error=nested.max_chord_error,
                    max_surface_segments=nested.max_surface_segments,
                )
            )

    def test_near_cylinder_scale_does_not_accept_one_percent_wrong_sphere(self) -> None:
        for progress in (1.0 - 1.0e-8, 1.0 - 1.0e-12):
            with self.subTest(progress=progress):
                request = _request(progress)
                surfaces = tuple(
                    SphereSpec(
                        item.surface_id,
                        item.center,
                        item.radius * 1.01,
                    )
                    if isinstance(item, SphereSpec)
                    and item.surface_id == "sphere:+1"
                    else item
                    for item in request.surfaces
                )
                with self.assertRaisesRegex(
                    NestedTangentCompositingError,
                    "not internally tangent",
                ):
                    compute_scene_occlusion_frame(
                        SceneOcclusionRequest(
                            request.scene_id,
                            surfaces,
                            request.view,
                            request.boundary_sources,
                            request.section,
                            request.tangent_spheres,
                            request.paint_policy,
                            max_chord_error=request.max_chord_error,
                            max_surface_segments=request.max_surface_segments,
                        )
                    )

    def test_remote_cylinder_trim_does_not_relax_local_tangency(self) -> None:
        request = _request(1.0)
        surfaces = tuple(
            CylinderSpec(
                item.surface_id,
                item.origin,
                item.axis,
                item.radius,
                (item.axial_range[0], 5.0e4),
                radial_axis=item.radial_axis,
            )
            if isinstance(item, CylinderSpec)
            else SphereSpec(
                item.surface_id,
                item.center,
                item.radius * 1.01,
            )
            if isinstance(item, SphereSpec) and item.surface_id == "sphere:+1"
            else item
            for item in request.surfaces
        )

        with self.assertRaisesRegex(
            NestedTangentCompositingError,
            "not internally tangent",
        ):
            compute_scene_occlusion_frame(
                SceneOcclusionRequest(
                    request.scene_id,
                    surfaces,
                    request.view,
                    request.boundary_sources,
                    request.section,
                    request.tangent_spheres,
                    request.paint_policy,
                    max_chord_error=request.max_chord_error,
                    max_surface_segments=request.max_surface_segments,
                )
            )

    def test_large_translation_does_not_relax_scaled_contact_certificates(self) -> None:
        request = _request(1.0)
        scale = 1.0e-6
        translation = np.asarray((1.0e6, -2.0e6, 3.0e6), dtype=float)

        def point(value):
            result = translation + scale * np.asarray(value, dtype=float)
            return tuple(float(item) for item in result)

        def vector(value):
            result = scale * np.asarray(value, dtype=float)
            return tuple(float(item) for item in result)

        original_mother = next(
            item for item in request.surfaces if isinstance(item, CylinderSpec)
        )
        mother = replace(
            original_mother,
            origin=point(original_mother.origin),
            radius=scale * original_mother.radius,
            axial_range=tuple(
                scale * value for value in original_mother.axial_range
            ),
        )
        original_sphere = next(
            item
            for item in request.surfaces
            if isinstance(item, SphereSpec) and item.surface_id == "sphere:+1"
        )
        sphere = replace(
            original_sphere,
            center=point(original_sphere.center),
            radius=scale * original_sphere.radius,
        )
        plane = replace(
            request.section.plane,
            point=point(request.section.plane.point),
        )
        original_source = next(
            item
            for item in request.boundary_sources
            if item.source_id == "contact:+1"
        )
        source = replace(
            original_source,
            curve=replace(
                original_source.curve,
                center=point(original_source.curve.center),
                first_axis=vector(original_source.curve.first_axis),
                second_axis=vector(original_source.curve.second_axis),
            ),
        )
        binding = next(
            item
            for item in request.tangent_spheres
            if item.sphere_surface_id == sphere.surface_id
        )
        context = GeometryContext().resolve(mother.characteristic_points)
        _certify_contact(
            mother,
            sphere,
            plane,
            source,
            binding,
            request.view,
            context,
        )

        with self.assertRaisesRegex(
            NestedTangentCompositingError,
            "not internally tangent",
        ):
            _certify_contact(
                mother,
                replace(sphere, radius=sphere.radius * 1.01),
                plane,
                source,
                binding,
                request.view,
                context,
            )

        wrong_radius_source = replace(
            source,
            curve=replace(
                source.curve,
                first_axis=tuple(
                    1.01 * value for value in source.curve.first_axis
                ),
                second_axis=tuple(
                    1.01 * value for value in source.curve.second_axis
                ),
            ),
        )
        with self.assertRaisesRegex(
            NestedTangentCompositingError,
            "wrong contact-circle radius",
        ):
            _certify_contact(
                mother,
                sphere,
                plane,
                wrong_radius_source,
                binding,
                request.view,
                context,
            )

        radial_shift = 0.01 * np.asarray(
            source.curve.first_axis,
            dtype=float,
        )
        shifted_source = replace(
            source,
            curve=replace(
                source.curve,
                center=tuple(
                    float(value)
                    for value in (
                        np.asarray(source.curve.center, dtype=float)
                        + radial_shift
                    )
                ),
            ),
        )
        with self.assertRaisesRegex(
            NestedTangentCompositingError,
            "wrong contact-circle center",
        ):
            _certify_contact(
                mother,
                sphere,
                plane,
                shifted_source,
                binding,
                request.view,
                context,
            )

        shifted_plane = replace(
            plane,
            point=tuple(
                float(value)
                for value in (
                    np.asarray(plane.point, dtype=float)
                    + 0.01
                    * sphere.radius
                    * np.asarray(plane.normal, dtype=float)
                )
            ),
        )
        with self.assertRaisesRegex(
            NestedTangentCompositingError,
            "not tangent to the cutting plane",
        ):
            _certify_contact(
                mother,
                sphere,
                shifted_plane,
                source,
                binding,
                request.view,
                context,
            )

    def test_edge_on_single_section_preserves_rank_one_curve_evidence(self) -> None:
        surface = SphereSpec("edge-sphere", (0.0, 0.0, 0.0), 2.0)
        plane = SectionPlane(
            "edge-plane",
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
        patch = PlaneDisplayPatchSpec(
            "edge-patch",
            plane.plane_id,
            2.5,
            2.5,
        )
        view = ParallelView.from_matrix(np.eye(3))
        curves = compute_quadric_section_boundary_curves(
            "edge-section",
            surface,
            plane,
        )
        sources = tuple(
            section_curve_boundary_source(
                curve,
                surface,
                plane,
                section_id="edge-section",
                authoritative_curves=curves,
            )
            for curve in curves
        )
        frame = compute_scene_occlusion_frame(
            SceneOcclusionRequest(
                "edge-on-single",
                (surface,),
                view,
                sources,
                SceneSectionSpec(surface.surface_id, plane, patch),
                paint_policy=QuadricPaintPolicy.DEPTH_AWARE_DIAGRAMMATIC,
            )
        )
        self.assertIs(frame.dispatch_path, SceneOcclusionPath.SINGLE_SECTION)
        self.assertIs(
            frame.section_frame.projection_kind,
            PlanePatchProjectionKind.LINE,
        )
        self.assertIsNotNone(frame.boundary_frame.rank_one_section_source_group)
        self.assertEqual(
            frame.boundary_frame.rank_one_section_source_group.source_ids,
            tuple(item.source_id for item in sources),
        )

    def test_near_cylinder_keeps_stable_semantic_sources(self) -> None:
        near = _frame(0.9999)
        cylinder = _frame(1.0)
        self.assertEqual(
            tuple(item.source_id for item in near.boundary_frame.sources),
            tuple(item.source_id for item in cylinder.boundary_frame.sources),
        )
        self.assertEqual(
            tuple(item.sphere_surface_id for item in near.nested_parent_frame.contacts),
            tuple(
                item.sphere_surface_id
                for item in cylinder.nested_parent_frame.contacts
            ),
        )

    def test_nested_diagrammatic_policy_keeps_hidden_curves_as_overlays(self) -> None:
        request = _request(0.5)
        frame = compute_scene_occlusion_frame(
            SceneOcclusionRequest(
                request.scene_id,
                request.surfaces,
                request.view,
                request.boundary_sources,
                request.section,
                request.tangent_spheres,
                QuadricPaintPolicy.DIAGRAMMATIC,
                max_chord_error=request.max_chord_error,
                max_surface_segments=request.max_surface_segments,
            )
        )
        rank = {item_id: index for index, item_id in enumerate(frame.draw_order)}
        contacts = {
            item.contact_source_id: item
            for item in frame.nested_parent_frame.contacts
        }
        for fragment in frame.boundary_frame.painted_fragments:
            contact = contacts.get(fragment.source_id)
            if contact is not None:
                self.assertLess(
                    rank[contact.sphere_item_id],
                    rank[fragment.item_id],
                )

    def test_authoritative_frames_reject_forged_reverse_draw_orders(self) -> None:
        frame = _frame(0.5)
        with self.assertRaisesRegex(
            SceneOcclusionError,
            "violates a painter relation",
        ):
            replace(frame, draw_order=tuple(reversed(frame.draw_order)))
        with self.assertRaisesRegex(
            NestedTangentCompositingError,
            "violates a painter relation",
        ):
            replace(
                frame.nested_parent_frame,
                draw_order=tuple(
                    reversed(frame.nested_parent_frame.draw_order)
                ),
            )

    def test_dispatch_path_cannot_be_forged_away_from_nested_evidence(self) -> None:
        frame = _frame(0.5)
        for path in (
            SceneOcclusionPath.GLOBAL_DISJOINT,
            SceneOcclusionPath.SINGLE_SECTION,
        ):
            with self.subTest(path=path.value):
                with self.assertRaises(SceneOcclusionError):
                    replace(frame, dispatch_path=path)
        with self.assertRaisesRegex(
            SceneOcclusionError,
            "physical surface authority",
        ):
            replace(frame, physical_surface_visibility_authoritative=True)

    def test_nested_contact_and_surface_evidence_cannot_be_forged(self) -> None:
        frame = _frame(0.5)
        parent = frame.nested_parent_frame
        first = parent.contacts[0]
        second = parent.contacts[1]

        with self.assertRaisesRegex(
            NestedTangentCompositingError,
            "unique sorted spheres",
        ):
            replace(parent, contacts=(first, first))
        with self.assertRaisesRegex(
            NestedTangentCompositingError,
            "surface painter mapping",
        ):
            replace(
                parent,
                surface_item_by_id=tuple(
                    (surface_id, first.sphere_item_id)
                    if surface_id == second.sphere_surface_id
                    else (surface_id, item_id)
                    for surface_id, item_id in parent.surface_item_by_id
                ),
            )

        forged_contact = replace(first, contact_source_id="forged-source")
        forged_parent = replace(
            parent,
            contacts=(forged_contact, second),
        )
        with self.assertRaisesRegex(
            SceneOcclusionError,
            "disagrees with boundary sources",
        ):
            replace(frame, nested_parent_frame=forged_parent)

    def test_boundary_frame_cannot_hide_mismatched_parent_evidence(self) -> None:
        request = _request(0.5)
        nested = _frame(0.5)
        other_mother, _records = _geometry(0.25)
        other_global = compute_global_quadric_frame(
            (),
            (other_mother,),
            request.view,
            paint_policy=request.paint_policy,
            max_chord_error=request.max_chord_error,
            max_segments=request.max_surface_segments,
        )

        with self.assertRaisesRegex(
            SceneOcclusionError,
            "derive from the selected global frame",
        ):
            replace(nested, global_frame=other_global)

        boundary = nested.boundary_frame
        removed = next(
            item
            for item in nested.nested_parent_frame.order_relations
            if item.reason == "plane_role_in_front_of_sphere"
        )
        forged_boundary = replace(
            boundary,
            order_relations=tuple(
                item
                for item in boundary.order_relations
                if (item.far_item_id, item.near_item_id)
                != (removed.far_item_id, removed.near_item_id)
            ),
        )
        with self.assertRaisesRegex(
            SceneOcclusionError,
            "omits a selected parent painter relation",
        ):
            replace(
                nested,
                boundary_frame=forged_boundary,
                order_relations=forged_boundary.order_relations,
            )

    def test_nested_parent_evidence_is_bound_to_plane_view_and_skeleton(self) -> None:
        request = _request(0.5)
        frame = _frame(0.5)
        parent = frame.nested_parent_frame
        mother = next(
            item for item in request.surfaces if item.surface_id == "mother"
        )
        spheres = tuple(
            item for item in request.surfaces if isinstance(item, SphereSpec)
        )
        other_plane = SectionPlane(
            "other-plane",
            request.section.plane.point,
            request.section.plane.normal,
            request.section.plane.u_axis,
        )
        common = {
            "context": frame.global_frame.geometry_context,
            "max_chord_error": request.max_chord_error,
            "max_surface_segments": request.max_surface_segments,
        }

        with self.assertRaisesRegex(
            NestedTangentCompositingError,
            "supplied cutting plane",
        ):
            compute_nested_tangent_parent_frame(
                mother,
                spheres,
                other_plane,
                frame.section_frame,
                request.boundary_sources,
                request.tangent_spheres,
                request.view,
                **common,
            )

        rotated_screen_view = ParallelView.from_matrix(
            np.vstack((-0.65 * _RIGHT, -0.65 * _UP, _DEPTH))
        )
        with self.assertRaisesRegex(
            NestedTangentCompositingError,
            "supplied parallel view",
        ):
            compute_nested_tangent_parent_frame(
                mother,
                spheres,
                request.section.plane,
                frame.section_frame,
                request.boundary_sources,
                request.tangent_spheres,
                rotated_screen_view,
                **common,
            )

        alternate_pair = compute_global_quadric_frame(
            (),
            spheres,
            rotated_screen_view,
            context=frame.global_frame.geometry_context,
            paint_policy=QuadricPaintPolicy.PHYSICAL,
            max_chord_error=request.max_chord_error,
            max_segments=request.max_surface_segments,
        )
        alternate_parent = replace(
            parent,
            sphere_pair_frame=alternate_pair,
        )
        with self.assertRaisesRegex(
            SceneOcclusionError,
            "share the selected view and context",
        ):
            replace(frame, nested_parent_frame=alternate_parent)

        reverse_view = ParallelView.from_matrix(
            np.vstack((0.65 * _RIGHT, -0.65 * _UP, -_DEPTH))
        )
        reverse_pair = compute_global_quadric_frame(
            (),
            spheres,
            reverse_view,
            context=frame.global_frame.geometry_context,
            paint_policy=QuadricPaintPolicy.PHYSICAL,
            max_chord_error=request.max_chord_error,
            max_segments=request.max_surface_segments,
        )
        with self.assertRaisesRegex(
            NestedTangentCompositingError,
            "missing from parent relations",
        ):
            replace(parent, sphere_pair_frame=reverse_pair)

        contact = parent.contacts[0]
        mandatory_pairs = {
            (
                frame.section_frame.paint_items.surface_back,
                contact.sphere_item_id,
            ),
            (
                contact.sphere_item_id,
                frame.section_frame.paint_items.surface_front,
            ),
        }
        shrunken_parent = replace(
            parent,
            order_relations=tuple(
                item
                for item in parent.order_relations
                if (item.far_item_id, item.near_item_id)
                not in mandatory_pairs
            ),
        )
        with self.assertRaisesRegex(
            SceneOcclusionError,
            "omits a mandatory painter relation",
        ):
            replace(frame, nested_parent_frame=shrunken_parent)

    def test_mother_hidden_curve_in_front_of_sphere_paints_after_sphere(self) -> None:
        request = _request(0.5)
        mother = next(
            item
            for item in request.surfaces
            if item.surface_id == "mother"
        )
        sphere = next(
            item
            for item in request.surfaces
            if item.surface_id == "sphere:+1"
        )
        direction = np.asarray(request.view.view_direction, dtype=float)
        mother_parameters = tuple(
            item.parameter
            for item in mother.ray_hits(
                sphere.center,
                direction,
                forward_only=True,
            )
            if item.parameter > sphere.radius
        )
        self.assertTrue(mother_parameters)
        parameter = 0.5 * (sphere.radius + min(mother_parameters))
        midpoint = np.asarray(sphere.center, dtype=float) + parameter * direction
        displacement = 0.01 * _RIGHT
        curve = SegmentCurve(
            "between-sphere-and-mother",
            tuple(float(item) for item in midpoint - displacement),
            tuple(float(item) for item in midpoint + displacement),
        )
        source = curve_boundary_source(curve, style_id="style:between")
        frame = compute_scene_occlusion_frame(
            SceneOcclusionRequest(
                request.scene_id,
                request.surfaces,
                request.view,
                (*request.boundary_sources, source),
                request.section,
                request.tangent_spheres,
                request.paint_policy,
                max_chord_error=request.max_chord_error,
                max_surface_segments=request.max_surface_segments,
            )
        )
        fragments = tuple(
            item
            for item in frame.boundary_frame.painted_fragments
            if item.source_id == source.source_id
            and item.occluder_surface_ids == (mother.surface_id,)
        )
        self.assertTrue(fragments)
        rank = {item_id: index for index, item_id in enumerate(frame.draw_order)}
        sphere_item = frame.nested_parent_frame.surface_items[sphere.surface_id]
        for fragment in fragments:
            self.assertLess(rank[sphere_item], rank[fragment.item_id])
            self.assertLess(
                rank[fragment.item_id],
                rank[frame.section_frame.paint_items.surface_front],
            )


if __name__ == "__main__":
    unittest.main()
