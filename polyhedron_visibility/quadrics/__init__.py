"""Analytic quadratic surfaces, conic sections, and curve visibility.

The public surface is loaded lazily so importing the renderer-neutral geometry
stack never imports Manim.  Renderer bindings are added by their own module and
remain optional until explicitly requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Final


_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "AffineFrame3D": (".algebra", "AffineFrame3D"),
    "CoincidentRayError": (".algebra", "CoincidentRayError"),
    "HomogeneousQuadric": (".algebra", "HomogeneousQuadric"),
    "QuadricAlgebraError": (".algebra", "QuadricAlgebraError"),
    "ConicClassification": (".conics", "ConicClassification"),
    "ConicError": (".conics", "ConicError"),
    "ConicKind": (".conics", "ConicKind"),
    "ConicParameterization": (".conics", "ConicParameterization"),
    "classify_conic": (".conics", "classify_conic"),
    "CircularTrimRimSpec": (".contract", "CircularTrimRimSpec"),
    "ConeModel": (".contract", "ConeModel"),
    "ConeSpec": (".contract", "ConeSpec"),
    "CylinderModel": (".contract", "CylinderModel"),
    "CylinderSpec": (".contract", "CylinderSpec"),
    "PlanarCapSpec": (".contract", "PlanarCapSpec"),
    "PlaneDisplayPatchSpec": (".contract", "PlaneDisplayPatchSpec"),
    "QuadricContractError": (".contract", "QuadricContractError"),
    "QuadricRayHit": (".contract", "QuadricRayHit"),
    "SectionPlane": (".contract", "SectionPlane"),
    "SphereSpec": (".contract", "SphereSpec"),
    "ANALYTIC_CURVE_SCHEMA": (".curves", "ANALYTIC_CURVE_SCHEMA"),
    "POINT_MARKER_SCHEMA": (".curves", "POINT_MARKER_SCHEMA"),
    "CircleArcCurve": (".curves", "CircleArcCurve"),
    "CurveContractError": (".curves", "CurveContractError"),
    "EllipseArcCurve": (".curves", "EllipseArcCurve"),
    "ParametricConicBranch": (".curves", "ParametricConicBranch"),
    "PointMarker3D": (".curves", "PointMarker3D"),
    "SegmentCurve": (".curves", "SegmentCurve"),
    "PLANAR_FRAME_3D_SCHEMA": (".planar_curves", "PLANAR_FRAME_3D_SCHEMA"),
    "PLANAR_POINT_3D_SCHEMA": (".planar_curves", "PLANAR_POINT_3D_SCHEMA"),
    "PLANAR_CURVE_3D_SCHEMA": (".planar_curves", "PLANAR_CURVE_3D_SCHEMA"),
    "PLANAR_CURVE_SCENE_3D_SCHEMA": (
        ".planar_curves",
        "PLANAR_CURVE_SCENE_3D_SCHEMA",
    ),
    "PlanarCurve3DContractError": (
        ".planar_curves",
        "PlanarCurve3DContractError",
    ),
    "PlanarFrame3D": (".planar_curves", "PlanarFrame3D"),
    "PlanarPoint3D": (".planar_curves", "PlanarPoint3D"),
    "Circle3DSpec": (".planar_curves", "Circle3DSpec"),
    "Ellipse3DSpec": (".planar_curves", "Ellipse3DSpec"),
    "PlanarCurve3DSpec": (".planar_curves", "PlanarCurve3DSpec"),
    "PlanarCurveScene3D": (".planar_curves", "PlanarCurveScene3D"),
    "DANDELIN_CONSTRUCTION_SCHEMA": (
        ".dandelin",
        "DANDELIN_CONSTRUCTION_SCHEMA",
    ),
    "DANDELIN_DIRECTRIX_SCHEMA": (
        ".dandelin",
        "DANDELIN_DIRECTRIX_SCHEMA",
    ),
    "DANDELIN_SPHERE_SCHEMA": (".dandelin", "DANDELIN_SPHERE_SCHEMA"),
    "DandelinConicFamily": (".dandelin", "DandelinConicFamily"),
    "DandelinConstruction3D": (".dandelin", "DandelinConstruction3D"),
    "DandelinConstructionError": (
        ".dandelin",
        "DandelinConstructionError",
    ),
    "DandelinDirectrix3D": (".dandelin", "DandelinDirectrix3D"),
    "DandelinPlaneSide": (".dandelin", "DandelinPlaneSide"),
    "DandelinSphere3D": (".dandelin", "DandelinSphere3D"),
    "canonical_dandelin_construction_json": (
        ".dandelin",
        "canonical_dandelin_construction_json",
    ),
    "compute_dandelin_construction": (
        ".dandelin",
        "compute_dandelin_construction",
    ),
    "DANDELIN_MERIDIAN_DIAGRAM_2D_SCHEMA": (
        ".dandelin_views",
        "DANDELIN_MERIDIAN_DIAGRAM_2D_SCHEMA",
    ),
    "DANDELIN_SECTION_PLANE_DIAGRAM_2D_SCHEMA": (
        ".dandelin_views",
        "DANDELIN_SECTION_PLANE_DIAGRAM_2D_SCHEMA",
    ),
    "DandelinCircleLineTangency2D": (
        ".dandelin_views",
        "DandelinCircleLineTangency2D",
    ),
    "DandelinDiagramLine2D": (".dandelin_views", "DandelinDiagramLine2D"),
    "DandelinDiagramPoint2D": (".dandelin_views", "DandelinDiagramPoint2D"),
    "DandelinDiagramSegment2D": (
        ".dandelin_views",
        "DandelinDiagramSegment2D",
    ),
    "DandelinMeridianDiagram2D": (
        ".dandelin_views",
        "DandelinMeridianDiagram2D",
    ),
    "DandelinSectionPlaneDiagram2D": (
        ".dandelin_views",
        "DandelinSectionPlaneDiagram2D",
    ),
    "DandelinSphereCircleSection2D": (
        ".dandelin_views",
        "DandelinSphereCircleSection2D",
    ),
    "DandelinSpherePlaneTangency2D": (
        ".dandelin_views",
        "DandelinSpherePlaneTangency2D",
    ),
    "DandelinView2DError": (".dandelin_views", "DandelinView2DError"),
    "build_dandelin_meridian_diagram": (
        ".dandelin_views",
        "build_dandelin_meridian_diagram",
    ),
    "build_dandelin_section_plane_diagram": (
        ".dandelin_views",
        "build_dandelin_section_plane_diagram",
    ),
    "canonical_dandelin_meridian_diagram_json": (
        ".dandelin_views",
        "canonical_dandelin_meridian_diagram_json",
    ),
    "canonical_dandelin_section_plane_diagram_json": (
        ".dandelin_views",
        "canonical_dandelin_section_plane_diagram_json",
    ),
    "DANDELIN_TEACHING_OVERLAY_SCHEMA": (
        ".dandelin_overlay",
        "DANDELIN_TEACHING_OVERLAY_SCHEMA",
    ),
    "DandelinTeachingOverlay3D": (
        ".dandelin_overlay",
        "DandelinTeachingOverlay3D",
    ),
    "DandelinTeachingOverlayError": (
        ".dandelin_overlay",
        "DandelinTeachingOverlayError",
    ),
    "build_dandelin_teaching_overlay": (
        ".dandelin_overlay",
        "build_dandelin_teaching_overlay",
    ),
    "canonical_dandelin_teaching_overlay_json": (
        ".dandelin_overlay",
        "canonical_dandelin_teaching_overlay_json",
    ),
    "DANDELIN_TANGENT_CONTACT_SCHEMA": (
        ".dandelin_visibility",
        "DANDELIN_TANGENT_CONTACT_SCHEMA",
    ),
    "DANDELIN_VISIBILITY_FRAME_SCHEMA": (
        ".dandelin_visibility",
        "DANDELIN_VISIBILITY_FRAME_SCHEMA",
    ),
    "DandelinTangentContactEvidence": (
        ".dandelin_visibility",
        "DandelinTangentContactEvidence",
    ),
    "DandelinVisibilityError": (
        ".dandelin_visibility",
        "DandelinVisibilityError",
    ),
    "DandelinVisibilityFrame": (
        ".dandelin_visibility",
        "DandelinVisibilityFrame",
    ),
    "DandelinVisibilitySource": (
        ".dandelin_visibility",
        "DandelinVisibilitySource",
    ),
    "DandelinVisibilityStroke": (
        ".dandelin_visibility",
        "DandelinVisibilityStroke",
    ),
    "certify_dandelin_tangent_contacts": (
        ".dandelin_visibility",
        "certify_dandelin_tangent_contacts",
    ),
    "canonical_dandelin_visibility_json": (
        ".dandelin_visibility",
        "canonical_dandelin_visibility_json",
    ),
    "build_dandelin_visibility_sources": (
        ".dandelin_visibility",
        "build_dandelin_visibility_sources",
    ),
    "compute_dandelin_visibility_frame": (
        ".dandelin_visibility",
        "compute_dandelin_visibility_frame",
    ),
    "fit_dandelin_visibility_patch": (
        ".dandelin_visibility",
        "fit_dandelin_visibility_patch",
    ),
    "DEFAULT_DANDELIN_BOUNDARY_STYLES": (
        ".dandelin_manim",
        "DEFAULT_DANDELIN_BOUNDARY_STYLES",
    ),
    "DEFAULT_DANDELIN_OCCLUSION_LIMITS": (
        ".dandelin_manim",
        "DEFAULT_DANDELIN_OCCLUSION_LIMITS",
    ),
    "DEFAULT_DANDELIN_OCCLUSION_STYLE": (
        ".dandelin_manim",
        "DEFAULT_DANDELIN_OCCLUSION_STYLE",
    ),
    "DandelinManimError": (
        ".dandelin_manim",
        "DandelinManimError",
    ),
    "DandelinOcclusion3D": (
        ".dandelin_manim",
        "DandelinOcclusion3D",
    ),
    "DANDELIN_SURFACE_LAYER_FRAME_SCHEMA": (
        ".dandelin_compositing",
        "DANDELIN_SURFACE_LAYER_FRAME_SCHEMA",
    ),
    "DandelinConeLayer": (
        ".dandelin_compositing",
        "DandelinConeLayer",
    ),
    "DandelinContactSheet": (
        ".dandelin_compositing",
        "DandelinContactSheet",
    ),
    "DandelinContactSheetSpan": (
        ".dandelin_compositing",
        "DandelinContactSheetSpan",
    ),
    "DandelinEqualDepthContact": (
        ".dandelin_compositing",
        "DandelinEqualDepthContact",
    ),
    "DandelinPlaneLayer": (
        ".dandelin_compositing",
        "DandelinPlaneLayer",
    ),
    "DandelinPlaneOutlineLayer": (
        ".dandelin_compositing",
        "DandelinPlaneOutlineLayer",
    ),
    "DandelinPlanePosition": (
        ".dandelin_compositing",
        "DandelinPlanePosition",
    ),
    "DandelinSphereLayer": (
        ".dandelin_compositing",
        "DandelinSphereLayer",
    ),
    "DandelinSpherePairEvidence": (
        ".dandelin_compositing",
        "DandelinSpherePairEvidence",
    ),
    "DandelinSurfaceCompositingError": (
        ".dandelin_compositing",
        "DandelinSurfaceCompositingError",
    ),
    "DandelinSurfaceLayerFrame": (
        ".dandelin_compositing",
        "DandelinSurfaceLayerFrame",
    ),
    "canonical_dandelin_surface_layer_json": (
        ".dandelin_compositing",
        "canonical_dandelin_surface_layer_json",
    ),
    "compute_dandelin_surface_layer_frame": (
        ".dandelin_compositing",
        "compute_dandelin_surface_layer_frame",
    ),
    "DEFAULT_DANDELIN_OVERLAY_STYLE": (
        ".dandelin_authoring",
        "DEFAULT_DANDELIN_OVERLAY_STYLE",
    ),
    "DEFAULT_DANDELIN_SECTION_STYLE": (
        ".dandelin_authoring",
        "DEFAULT_DANDELIN_SECTION_STYLE",
    ),
    "DandelinSection3D": (".dandelin_authoring", "DandelinSection3D"),
    "DandelinSectionAuthoringError": (
        ".dandelin_authoring",
        "DandelinSectionAuthoringError",
    ),
    "CRITICAL_EVENT_SCHEMA": (".critical", "CRITICAL_EVENT_SCHEMA"),
    "CriticalEvidence": (".critical", "CriticalEvidence"),
    "CriticalEvent": (".critical", "CriticalEvent"),
    "CriticalEventError": (".critical", "CriticalEventError"),
    "CriticalEventKind": (".critical", "CriticalEventKind"),
    "compute_curve_critical_events": (
        ".critical",
        "compute_curve_critical_events",
    ),
    "AnalyticCurve3D": (".critical", "AnalyticCurve3D"),
    "QuadricSurfaceSpec": (".sections", "QuadricSurfaceSpec"),
    "MAX_POLYNOMIAL_DEGREE": (".roots", "MAX_POLYNOMIAL_DEGREE"),
    "ExpChartRoot": (".roots", "ExpChartRoot"),
    "PolynomialRootError": (".roots", "PolynomialRootError"),
    "RealRoot": (".roots", "RealRoot"),
    "cluster_real_roots": (".roots", "cluster_real_roots"),
    "solve_real_polynomial": (".roots", "solve_real_polynomial"),
    "solve_real_polynomial_exp_chart": (
        ".roots",
        "solve_real_polynomial_exp_chart",
    ),
    "OPAQUE_PROJECTION_PROXY_SCHEMA": (
        ".projection",
        "OPAQUE_PROJECTION_PROXY_SCHEMA",
    ),
    "CONE_PROJECTION_LAYERS_SCHEMA": (
        ".projection",
        "CONE_PROJECTION_LAYERS_SCHEMA",
    ),
    "ConeProjectionLayers": (".projection", "ConeProjectionLayers"),
    "ConeProjectionSheet": (".projection", "ConeProjectionSheet"),
    "OpaqueProjectionProxy": (".projection", "OpaqueProjectionProxy"),
    "ParallelViewInput": (".projection", "ParallelViewInput"),
    "ProjectionApproximationMetadata": (
        ".projection",
        "ProjectionApproximationMetadata",
    ),
    "ProjectionProxyError": (".projection", "ProjectionProxyError"),
    "ProjectionSubdivisionError": (
        ".projection",
        "ProjectionSubdivisionError",
    ),
    "build_opaque_projection_proxy": (
        ".projection",
        "build_opaque_projection_proxy",
    ),
    "build_cone_projection_layers": (
        ".projection",
        "build_cone_projection_layers",
    ),
    "canonical_opaque_projection_proxy_json": (
        ".projection",
        "canonical_opaque_projection_proxy_json",
    ),
    "QuadricSectionError": (".sections", "QuadricSectionError"),
    "FiniteSectionBoundaryCurve": (
        ".sections",
        "FiniteSectionBoundaryCurve",
    ),
    "QuadricSectionBoundary": (".sections", "QuadricSectionBoundary"),
    "UnboundedFiniteSectionError": (".sections", "UnboundedFiniteSectionError"),
    "compute_quadric_section": (".sections", "compute_quadric_section"),
    "compute_quadric_section_boundary": (
        ".sections",
        "compute_quadric_section_boundary",
    ),
    "compute_quadric_section_boundary_curves": (
        ".sections",
        "compute_quadric_section_boundary_curves",
    ),
    "compute_section_cap_chord_curves": (
        ".sections",
        "compute_section_cap_chord_curves",
    ),
    "intersect_plane_with_quadric": (".sections", "intersect_plane_with_quadric"),
    "restrict_quadric_to_plane": (".sections", "restrict_quadric_to_plane"),
    "section_cap_chord_curve_ids": (
        ".sections",
        "section_cap_chord_curve_ids",
    ),
    "FiniteSectionTopology": (".trace", "FiniteSectionTopology"),
    "QUADRIC_SECTION_TRACE_SCHEMA": (
        ".trace",
        "QUADRIC_SECTION_TRACE_SCHEMA",
    ),
    "QuadricSectionTrace": (".trace", "QuadricSectionTrace"),
    "SectionBranchTrace": (".trace", "SectionBranchTrace"),
    "SectionComponentTrace": (".trace", "SectionComponentTrace"),
    "canonical_quadric_section_trace_json": (
        ".trace",
        "canonical_quadric_section_trace_json",
    ),
    "section_trace_curves": (".trace", "section_trace_curves"),
    "QUADRIC_VISIBILITY_FRAME_SCHEMA": (
        ".visibility",
        "QUADRIC_VISIBILITY_FRAME_SCHEMA",
    ),
    "QUADRIC_VISIBILITY_RECORD_SCHEMA": (
        ".visibility",
        "QUADRIC_VISIBILITY_RECORD_SCHEMA",
    ),
    "POINT_VISIBILITY_RECORD_SCHEMA": (
        ".visibility",
        "POINT_VISIBILITY_RECORD_SCHEMA",
    ),
    "CurveVisibilityFrame": (".visibility", "CurveVisibilityFrame"),
    "CurveVisibilityRecord": (".visibility", "CurveVisibilityRecord"),
    "PointVisibilityRecord": (".visibility", "PointVisibilityRecord"),
    "QuadricVisibilityError": (".visibility", "QuadricVisibilityError"),
    "canonical_quadric_visibility_json": (
        ".visibility",
        "canonical_quadric_visibility_json",
    ),
    "compute_curve_visibility": (".visibility", "compute_curve_visibility"),
    "compute_point_visibility": (".visibility", "compute_point_visibility"),
    "compute_quadric_visibility": (".visibility", "compute_quadric_visibility"),
    "DEFAULT_PLANE_PATCH_MARGIN_RATIO": (
        ".plane_patch",
        "DEFAULT_PLANE_PATCH_MARGIN_RATIO",
    ),
    "FittedPlaneDisplayPatch": (".plane_patch", "FittedPlaneDisplayPatch"),
    "PLANE_MOTION_PATCH_ENVELOPE_SCHEMA": (
        ".plane_patch",
        "PLANE_MOTION_PATCH_ENVELOPE_SCHEMA",
    ),
    "PLANE_PATCH_FIT_SCHEMA": (".plane_patch", "PLANE_PATCH_FIT_SCHEMA"),
    "PlaneMotionPatchEnvelope": (".plane_patch", "PlaneMotionPatchEnvelope"),
    "PlanePatchFitError": (".plane_patch", "PlanePatchFitError"),
    "SurfaceMotionRadius": (".plane_patch", "SurfaceMotionRadius"),
    "SurfacePlaneExtents": (".plane_patch", "SurfacePlaneExtents"),
    "canonical_fitted_plane_display_patch_json": (
        ".plane_patch",
        "canonical_fitted_plane_display_patch_json",
    ),
    "canonical_plane_motion_patch_envelope_json": (
        ".plane_patch",
        "canonical_plane_motion_patch_envelope_json",
    ),
    "finite_surface_support_interval": (
        ".plane_patch",
        "finite_surface_support_interval",
    ),
    "fit_plane_display_patch": (".plane_patch", "fit_plane_display_patch"),
    "fit_plane_motion_display_patch_envelope": (
        ".plane_patch",
        "fit_plane_motion_display_patch_envelope",
    ),
    "PROJECTED_CURVE_CROSSING_SCHEMA": (
        ".curve_intersections",
        "PROJECTED_CURVE_CROSSING_SCHEMA",
    ),
    "ProjectedCurveCrossing": (
        ".curve_intersections",
        "ProjectedCurveCrossing",
    ),
    "ProjectedCurveIntersectionError": (
        ".curve_intersections",
        "ProjectedCurveIntersectionError",
    ),
    "canonical_projected_curve_crossings_json": (
        ".curve_intersections",
        "canonical_projected_curve_crossings_json",
    ),
    "compute_projected_curve_crossings": (
        ".curve_intersections",
        "compute_projected_curve_crossings",
    ),
    "QUADRIC_COMPOSITING_FRAME_SCHEMA": (
        ".compositing",
        "QUADRIC_COMPOSITING_FRAME_SCHEMA",
    ),
    "QuadricCompositingError": (".compositing", "QuadricCompositingError"),
    "QuadricCompositingFrame": (".compositing", "QuadricCompositingFrame"),
    "QuadricCurvePaintFragment": (
        ".compositing",
        "QuadricCurvePaintFragment",
    ),
    "QuadricPaintKind": (".compositing", "QuadricPaintKind"),
    "QuadricPaintPolicy": (".compositing", "QuadricPaintPolicy"),
    "BoundaryOcclusionScope": (".boundary_compositing", "BoundaryOcclusionScope"),
    "BoundaryRenderIntent": (".boundary_compositing", "BoundaryRenderIntent"),
    "BoundaryScreenProjectionDimension": (
        ".boundary_compositing",
        "BoundaryScreenProjectionDimension",
    ),
    "BoundarySectionAnchors": (".boundary_compositing", "BoundarySectionAnchors"),
    "BoundaryPlaneRelation": (".boundary_section", "BoundaryPlaneRelation"),
    "QUADRIC_BOUNDARY_SECTION_LIMITS": (
        ".boundary_section",
        "QUADRIC_BOUNDARY_SECTION_LIMITS",
    ),
    "QuadricBoundarySectionLimits": (
        ".boundary_section",
        "QuadricBoundarySectionLimits",
    ),
    "QuadricBoundarySectionSpan": (
        ".boundary_section",
        "QuadricBoundarySectionSpan",
    ),
    "certify_rank_one_section_boundary_sources": (
        ".boundary_section",
        "certify_rank_one_section_boundary_sources",
    ),
    "compute_boundary_section_spans": (
        ".boundary_section",
        "compute_boundary_section_spans",
    ),
    "BoundarySemanticKind": (".boundary_compositing", "BoundarySemanticKind"),
    "BoundarySourceKind": (".boundary_compositing", "BoundarySourceKind"),
    "QUADRIC_BOUNDARY_COMPOSITING_SCHEMA": (
        ".boundary_compositing",
        "QUADRIC_BOUNDARY_COMPOSITING_SCHEMA",
    ),
    "QuadricBoundaryCompositingError": (
        ".boundary_compositing",
        "QuadricBoundaryCompositingError",
    ),
    "QuadricBoundaryCompositingFrame": (
        ".boundary_compositing",
        "QuadricBoundaryCompositingFrame",
    ),
    "QuadricBoundaryPaintFragment": (
        ".boundary_compositing",
        "QuadricBoundaryPaintFragment",
    ),
    "QuadricBoundarySource": (".boundary_compositing", "QuadricBoundarySource"),
    "QuadricBoundarySectionSourceProjection": (
        ".boundary_compositing",
        "QuadricBoundarySectionSourceProjection",
    ),
    "QuadricBoundaryVisibilitySpan": (
        ".boundary_compositing",
        "QuadricBoundaryVisibilitySpan",
    ),
    "QuadricRankOneSectionSourceGroup": (
        ".boundary_compositing",
        "QuadricRankOneSectionSourceGroup",
    ),
    "canonical_quadric_boundary_compositing_json": (
        ".boundary_compositing",
        "canonical_quadric_boundary_compositing_json",
    ),
    "compute_boundary_visibility": (
        ".boundary_compositing",
        "compute_boundary_visibility",
    ),
    "compute_quadric_boundary_crossings": (
        ".boundary_compositing",
        "compute_quadric_boundary_crossings",
    ),
    "compute_quadric_boundary_compositing": (
        ".boundary_compositing",
        "compute_quadric_boundary_compositing",
    ),
    "GeneratorBoundarySpec": (".surface_boundaries", "GeneratorBoundarySpec"),
    "SurfaceBoundarySlotDescriptor": (
        ".surface_boundaries",
        "SurfaceBoundarySlotDescriptor",
    ),
    "build_surface_boundary_sources": (
        ".surface_boundaries",
        "build_surface_boundary_sources",
    ),
    "curve_boundary_source": (".surface_boundaries", "curve_boundary_source"),
    "plane_outline_sources": (".surface_boundaries", "plane_outline_sources"),
    "section_curve_boundary_source": (
        ".surface_boundaries",
        "section_curve_boundary_source",
    ),
    "surface_boundary_source_ids": (
        ".surface_boundaries",
        "surface_boundary_source_ids",
    ),
    "surface_boundary_slot_descriptors": (
        ".surface_boundaries",
        "surface_boundary_slot_descriptors",
    ),
    "QuadricPaintRelation": (".compositing", "QuadricPaintRelation"),
    "QuadricStyleDescriptor": (".compositing", "QuadricStyleDescriptor"),
    "QuadricSurfacePaintItem": (".compositing", "QuadricSurfacePaintItem"),
    "canonical_quadric_compositing_json": (
        ".compositing",
        "canonical_quadric_compositing_json",
    ),
    "compute_quadric_compositing": (
        ".compositing",
        "compute_quadric_compositing",
    ),
    "PlaneDepthRole": (".section_compositing", "PlaneDepthRole"),
    "PLANE_PATCH_RANK_RATIO_THRESHOLD": (
        ".section_compositing",
        "PLANE_PATCH_RANK_RATIO_THRESHOLD",
    ),
    "PlanePatchProjectionEvidence": (
        ".section_compositing",
        "PlanePatchProjectionEvidence",
    ),
    "PlanePatchProjectionKind": (
        ".section_compositing",
        "PlanePatchProjectionKind",
    ),
    "QUADRIC_SECTION_COMPOSITING_LIMITS": (
        ".section_compositing",
        "QUADRIC_SECTION_COMPOSITING_LIMITS",
    ),
    "QUADRIC_SECTION_COMPOSITING_SCHEMA": (
        ".section_compositing",
        "QUADRIC_SECTION_COMPOSITING_SCHEMA",
    ),
    "QuadricPlaneFragment": (
        ".section_compositing",
        "QuadricPlaneFragment",
    ),
    "QuadricSectionCompositingError": (
        ".section_compositing",
        "QuadricSectionCompositingError",
    ),
    "QuadricSectionCompositingFrame": (
        ".section_compositing",
        "QuadricSectionCompositingFrame",
    ),
    "QuadricSectionCompositingLimits": (
        ".section_compositing",
        "QuadricSectionCompositingLimits",
    ),
    "QuadricSectionPaintItems": (
        ".section_compositing",
        "QuadricSectionPaintItems",
    ),
    "canonical_quadric_section_compositing_json": (
        ".section_compositing",
        "canonical_quadric_section_compositing_json",
    ),
    "compute_quadric_section_compositing": (
        ".section_compositing",
        "compute_quadric_section_compositing",
    ),
    "repaint_quadric_section_compositing": (
        ".section_compositing",
        "repaint_quadric_section_compositing",
    ),
    "quadric_plane_fragment_contours": (
        ".section_compositing",
        "quadric_plane_fragment_contours",
    ),
    "BranchCapacityPlan": (".animation", "BranchCapacityPlan"),
    "BranchContinuityError": (".animation", "BranchContinuityError"),
    "MAX_SECTION_BRANCH_SLOTS": (".animation", "MAX_SECTION_BRANCH_SLOTS"),
    "MovingPointContinuityError": (
        ".animation",
        "MovingPointContinuityError",
    ),
    "MovingPointSample": (".animation", "MovingPointSample"),
    "MovingPointTrace": (".animation", "MovingPointTrace"),
    "PointAuxiliaryRule": (".animation", "PointAuxiliaryRule"),
    "PointParameterMode": (".animation", "PointParameterMode"),
    "PointTrackSelection": (".animation", "PointTrackSelection"),
    "PointTransitionContext": (".animation", "PointTransitionContext"),
    "QUADRIC_SECTION_ANIMATION_SCHEMA": (
        ".animation",
        "QUADRIC_SECTION_ANIMATION_SCHEMA",
    ),
    "SectionAnimationError": (".animation", "SectionAnimationError"),
    "SectionAnimationSample": (".animation", "SectionAnimationSample"),
    "SectionAnimationTrace": (".animation", "SectionAnimationTrace"),
    "SectionConicFamily": (".animation", "SectionConicFamily"),
    "SectionTopologySignature": (".animation", "SectionTopologySignature"),
    "TopologyEvent": (".animation", "TopologyEvent"),
    "TopologyEventKind": (".animation", "TopologyEventKind"),
    "TrackedSectionBranch": (".animation", "TrackedSectionBranch"),
    "TrackedSectionFrame": (".animation", "TrackedSectionFrame"),
    "canonical_quadric_section_animation_json": (
        ".animation",
        "canonical_quadric_section_animation_json",
    ),
    "track_moving_section_point": (
        ".animation",
        "track_moving_section_point",
    ),
    "match_tracked_section_frame": (
        ".animation",
        "match_tracked_section_frame",
    ),
    "track_quadric_section_animation": (
        ".animation",
        "track_quadric_section_animation",
    ),
    "AxisAnglePlaneMotion": (".plane_motion", "AxisAnglePlaneMotion"),
    "PLANE_MOTION_SCHEDULE_SCHEMA": (
        ".plane_motion",
        "PLANE_MOTION_SCHEDULE_SCHEMA",
    ),
    "PlaneMotionCriticalEvent": (
        ".plane_motion",
        "PlaneMotionCriticalEvent",
    ),
    "PlaneMotionCriticalKind": (
        ".plane_motion",
        "PlaneMotionCriticalKind",
    ),
    "PlaneMotionError": (".plane_motion", "PlaneMotionError"),
    "PlaneMotionSchedule": (".plane_motion", "PlaneMotionSchedule"),
    "ScheduledSectionAnimation": (
        ".plane_motion",
        "ScheduledSectionAnimation",
    ),
    "canonical_plane_motion_schedule_json": (
        ".plane_motion",
        "canonical_plane_motion_schedule_json",
    ),
    "compute_plane_motion_schedule": (
        ".plane_motion",
        "compute_plane_motion_schedule",
    ),
    "track_scheduled_plane_section": (
        ".plane_motion",
        "track_scheduled_plane_section",
    ),
    "ParallelPlaneTranslation": (
        ".parallel_plane_motion",
        "ParallelPlaneTranslation",
    ),
    "ParallelPlaneMotionSchedule": (
        ".parallel_plane_motion",
        "ParallelPlaneMotionSchedule",
    ),
    "compute_parallel_plane_motion_schedule": (
        ".parallel_plane_motion",
        "compute_parallel_plane_motion_schedule",
    ),
    "SectionTimeline": (".section_timeline", "SectionTimeline"),
    "SectionTimelineError": (".section_timeline", "SectionTimelineError"),
    "compile_section_timeline": (
        ".section_timeline",
        "compile_section_timeline",
    ),
    "SectionDisplayCatalog": (
        ".semantic_display",
        "SectionDisplayCatalog",
    ),
    "SectionDisplayFrame": (".semantic_display", "SectionDisplayFrame"),
    "SectionDisplayInstruction": (
        ".semantic_display",
        "SectionDisplayInstruction",
    ),
    "SectionDisplayRole": (".semantic_display", "SectionDisplayRole"),
    "SectionSemanticSlot": (".semantic_display", "SectionSemanticSlot"),
    "compile_section_display": (
        ".semantic_display",
        "compile_section_display",
    ),
    "SECTION_COMPOSITING_FRAME_SCHEMA": (
        ".semantic_compositing",
        "SECTION_COMPOSITING_FRAME_SCHEMA",
    ),
    "SECTION_COMPOSITING_INSTRUCTION_SCHEMA": (
        ".semantic_compositing",
        "SECTION_COMPOSITING_INSTRUCTION_SCHEMA",
    ),
    "SectionCompositingAxes": (
        ".semantic_compositing",
        "SectionCompositingAxes",
    ),
    "SectionCompositingFrame": (
        ".semantic_compositing",
        "SectionCompositingFrame",
    ),
    "SectionCompositingInstruction": (
        ".semantic_compositing",
        "SectionCompositingInstruction",
    ),
    "SectionCompositingOverride": (
        ".semantic_compositing",
        "SectionCompositingOverride",
    ),
    "SectionCompositingSlotState": (
        ".semantic_compositing",
        "SectionCompositingSlotState",
    ),
    "SectionCompositingTargetKind": (
        ".semantic_compositing",
        "SectionCompositingTargetKind",
    ),
    "SectionDepthPresentationPolicy": (
        ".semantic_compositing",
        "SectionDepthPresentationPolicy",
    ),
    "SectionOcclusionParticipation": (
        ".semantic_compositing",
        "SectionOcclusionParticipation",
    ),
    "SectionSemanticCompositingError": (
        ".semantic_compositing",
        "SectionSemanticCompositingError",
    ),
    "compile_section_compositing": (
        ".semantic_compositing",
        "compile_section_compositing",
    ),
    "SectionTimelineTransitionPlan": (
        ".section_timeline_transition",
        "SectionTimelineTransitionPlan",
    ),
    "SectionTimelineTransitionState": (
        ".section_timeline_transition",
        "SectionTimelineTransitionState",
    ),
    "build_section_timeline_transition_plan": (
        ".section_timeline_transition",
        "build_section_timeline_transition_plan",
    ),
    "section_timeline_transition_state_at": (
        ".section_timeline_transition",
        "section_timeline_transition_state_at",
    ),
    "SECTION_TRANSITION_PLAN_SCHEMA": (
        ".transition",
        "SECTION_TRANSITION_PLAN_SCHEMA",
    ),
    "SectionTransitionError": (".transition", "SectionTransitionError"),
    "SectionTransitionFrame": (".transition", "SectionTransitionFrame"),
    "SectionTransitionLayer": (".transition", "SectionTransitionLayer"),
    "SectionTransitionMode": (".transition", "SectionTransitionMode"),
    "SectionTransitionPlan": (".transition", "SectionTransitionPlan"),
    "SectionTransitionRole": (".transition", "SectionTransitionRole"),
    "TopologyTransitionKnot": (".transition", "TopologyTransitionKnot"),
    "build_section_transition_plan": (
        ".transition",
        "build_section_transition_plan",
    ),
    "canonical_section_transition_plan_json": (
        ".transition",
        "canonical_section_transition_plan_json",
    ),
    "sample_section_transition": (
        ".transition",
        "sample_section_transition",
    ),
    "NESTED_TANGENT_CONTACT_EVIDENCE_SCHEMA": (
        ".nested_tangent_compositing",
        "NESTED_TANGENT_CONTACT_EVIDENCE_SCHEMA",
    ),
    "NESTED_TANGENT_PARENT_FRAME_SCHEMA": (
        ".nested_tangent_compositing",
        "NESTED_TANGENT_PARENT_FRAME_SCHEMA",
    ),
    "NestedTangentCompositingError": (
        ".nested_tangent_compositing",
        "NestedTangentCompositingError",
    ),
    "NestedTangentContactEvidence": (
        ".nested_tangent_compositing",
        "NestedTangentContactEvidence",
    ),
    "NestedTangentParentFrame": (
        ".nested_tangent_compositing",
        "NestedTangentParentFrame",
    ),
    "NestedTangentSphereSpec": (
        ".nested_tangent_compositing",
        "NestedTangentSphereSpec",
    ),
    "TangentSpherePlanePosition": (
        ".nested_tangent_compositing",
        "TangentSpherePlanePosition",
    ),
    "compute_nested_tangent_parent_frame": (
        ".nested_tangent_compositing",
        "compute_nested_tangent_parent_frame",
    ),
    "SCENE_OCCLUSION_FRAME_SCHEMA": (
        ".scene_occlusion",
        "SCENE_OCCLUSION_FRAME_SCHEMA",
    ),
    "SceneOcclusionCoordinator": (
        ".scene_occlusion",
        "SceneOcclusionCoordinator",
    ),
    "SceneOcclusionError": (".scene_occlusion", "SceneOcclusionError"),
    "SceneOcclusionFrame": (".scene_occlusion", "SceneOcclusionFrame"),
    "SceneOcclusionPath": (".scene_occlusion", "SceneOcclusionPath"),
    "SceneOcclusionRequest": (".scene_occlusion", "SceneOcclusionRequest"),
    "SceneSectionSpec": (".scene_occlusion", "SceneSectionSpec"),
    "canonical_scene_occlusion_json": (
        ".scene_occlusion",
        "canonical_scene_occlusion_json",
    ),
    "compute_scene_occlusion_frame": (
        ".scene_occlusion",
        "compute_scene_occlusion_frame",
    ),
    "GLOBAL_QUADRIC_FRAME_SCHEMA": (
        ".global_occlusion",
        "GLOBAL_QUADRIC_FRAME_SCHEMA",
    ),
    "GlobalQuadricFrame": (".global_occlusion", "GlobalQuadricFrame"),
    "GlobalQuadricOcclusionError": (
        ".global_occlusion",
        "GlobalQuadricOcclusionError",
    ),
    "StrictSeparationEvidence": (
        ".global_occlusion",
        "StrictSeparationEvidence",
    ),
    "SurfaceDepthEvidence": (".global_occlusion", "SurfaceDepthEvidence"),
    "SurfaceDepthWitness": (".global_occlusion", "SurfaceDepthWitness"),
    "SurfaceOrderConstraint": (
        ".global_occlusion",
        "SurfaceOrderConstraint",
    ),
    "canonical_global_quadric_frame_json": (
        ".global_occlusion",
        "canonical_global_quadric_frame_json",
    ),
    "compute_global_quadric_frame": (
        ".global_occlusion",
        "compute_global_quadric_frame",
    ),
    "verify_strict_quadric_separation": (
        ".global_occlusion",
        "verify_strict_quadric_separation",
    ),
    "PreparedQuadricManimFrame": (".manim", "PreparedQuadricManimFrame"),
    "DEFAULT_QUADRIC_VIEW": (".manim", "DEFAULT_QUADRIC_VIEW"),
    "QUADRIC_MANIM_LIMITS": (".manim", "QUADRIC_MANIM_LIMITS"),
    "QuadricManimCapacityError": (".manim", "QuadricManimCapacityError"),
    "QuadricManimError": (".manim", "QuadricManimError"),
    "QuadricManimLimits": (".manim", "QuadricManimLimits"),
    "QuadricManimStyle": (".manim", "QuadricManimStyle"),
    "QuadricBoundaryStyle": (".manim", "QuadricBoundaryStyle"),
    "QuadricGeometryPrototype": (".manim", "QuadricGeometryPrototype"),
    "QuadricOcclusion3D": (".manim", "QuadricOcclusion3D"),
    "QuadricOcclusionTransactionSnapshot": (
        ".manim",
        "QuadricOcclusionTransactionSnapshot",
    ),
    "estimate_quadric_mobject_count": (
        ".manim",
        "estimate_quadric_mobject_count",
    ),
    "QUADRIC_RENDER_PROFILE_SCHEMA": (
        ".profiles",
        "QUADRIC_RENDER_PROFILE_SCHEMA",
    ),
    "QUADRIC_PREVIEW_PROFILE": (".profiles", "QUADRIC_PREVIEW_PROFILE"),
    "QUADRIC_FINAL_PROFILE": (".profiles", "QUADRIC_FINAL_PROFILE"),
    "QUADRIC_RENDER_PROFILES": (".profiles", "QUADRIC_RENDER_PROFILES"),
    "QuadricRenderProfile": (".profiles", "QuadricRenderProfile"),
    "QUADRIC_CAPACITY_PLAN_SCHEMA": (
        ".capacity",
        "QUADRIC_CAPACITY_PLAN_SCHEMA",
    ),
    "QuadricCapacityHeadroom": (".capacity", "QuadricCapacityHeadroom"),
    "QuadricCapacityPeaks": (".capacity", "QuadricCapacityPeaks"),
    "QuadricCapacityPlan": (".capacity", "QuadricCapacityPlan"),
    "QuadricCapacityPlanner": (".capacity", "QuadricCapacityPlanner"),
    "QuadricCapacityPlanningError": (
        ".capacity",
        "QuadricCapacityPlanningError",
    ),
    "QuadricCapacitySample": (".capacity", "QuadricCapacitySample"),
    "canonical_quadric_capacity_plan_json": (
        ".capacity",
        "canonical_quadric_capacity_plan_json",
    ),
    "scheduled_capacity_progresses": (
        ".capacity",
        "scheduled_capacity_progresses",
    ),
    "PlaneInput": (".authoring", "PlaneInput"),
    "QuadricSection3D": (".authoring", "QuadricSection3D"),
    "CompositeQuadricSection3D": (
        ".composite_authoring",
        "CompositeQuadricSection3D",
    ),
    "CompositeQuadricSectionAuthoringError": (
        ".composite_authoring",
        "CompositeQuadricSectionAuthoringError",
    ),
    "PreparedCompositeQuadricSectionFrame": (
        ".composite_authoring",
        "PreparedCompositeQuadricSectionFrame",
    ),
    "COMPOSITE_QUADRIC_SECTION_COMPOSITING_SCHEMA": (
        ".composite_section",
        "COMPOSITE_QUADRIC_SECTION_COMPOSITING_SCHEMA",
    ),
    "CompositeQuadricSectionCompositingError": (
        ".composite_section",
        "CompositeQuadricSectionCompositingError",
    ),
    "CompositeQuadricSectionCompositingFrame": (
        ".composite_section",
        "CompositeQuadricSectionCompositingFrame",
    ),
    "CompositeQuadricSectionPaintItems": (
        ".composite_section",
        "CompositeQuadricSectionPaintItems",
    ),
    "CompositeSectionBranchLineage": (
        ".composite_section",
        "CompositeSectionBranchLineage",
    ),
    "CompositeSharedApexEvidence": (
        ".composite_section",
        "CompositeSharedApexEvidence",
    ),
    "CompositeSurfaceSheetItems": (
        ".composite_section",
        "CompositeSurfaceSheetItems",
    ),
    "canonical_composite_quadric_section_compositing_json": (
        ".composite_section",
        "canonical_composite_quadric_section_compositing_json",
    ),
    "compute_composite_quadric_section_compositing": (
        ".composite_section",
        "compute_composite_quadric_section_compositing",
    ),
    "QuadricSectionAuthoringError": (
        ".authoring",
        "QuadricSectionAuthoringError",
    ),
    "QuadricSectionAction": (".rig", "QuadricSectionAction"),
    "QuadricSectionRig": (".rig", "QuadricSectionRig"),
    "QuadricSectionRigError": (".rig", "QuadricSectionRigError"),
    "SectionState": (".rig", "SectionState"),
    "MAX_TRANSITION_INTERVAL_SLOTS": (
        ".transition_manim",
        "MAX_TRANSITION_INTERVAL_SLOTS",
    ),
    "PreparedSectionTransitionGeometry": (
        ".transition_manim",
        "PreparedSectionTransitionGeometry",
    ),
    "QuadricSectionTransition3D": (
        ".transition_manim",
        "QuadricSectionTransition3D",
    ),
    "QuadricSectionTransitionManimError": (
        ".transition_manim",
        "QuadricSectionTransitionManimError",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
