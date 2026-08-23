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
    "ConeSpec": (".contract", "ConeSpec"),
    "CylinderSpec": (".contract", "CylinderSpec"),
    "PlanarCapSpec": (".contract", "PlanarCapSpec"),
    "PlaneDisplayPatchSpec": (".contract", "PlaneDisplayPatchSpec"),
    "QuadricContractError": (".contract", "QuadricContractError"),
    "QuadricRayHit": (".contract", "QuadricRayHit"),
    "SectionPlane": (".contract", "SectionPlane"),
    "SphereSpec": (".contract", "SphereSpec"),
    "ANALYTIC_CURVE_SCHEMA": (".curves", "ANALYTIC_CURVE_SCHEMA"),
    "CircleArcCurve": (".curves", "CircleArcCurve"),
    "CurveContractError": (".curves", "CurveContractError"),
    "EllipseArcCurve": (".curves", "EllipseArcCurve"),
    "ParametricConicBranch": (".curves", "ParametricConicBranch"),
    "SegmentCurve": (".curves", "SegmentCurve"),
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
    "canonical_opaque_projection_proxy_json": (
        ".projection",
        "canonical_opaque_projection_proxy_json",
    ),
    "QuadricSectionError": (".sections", "QuadricSectionError"),
    "UnboundedFiniteSectionError": (".sections", "UnboundedFiniteSectionError"),
    "compute_quadric_section": (".sections", "compute_quadric_section"),
    "intersect_plane_with_quadric": (".sections", "intersect_plane_with_quadric"),
    "restrict_quadric_to_plane": (".sections", "restrict_quadric_to_plane"),
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
    "CurveVisibilityFrame": (".visibility", "CurveVisibilityFrame"),
    "CurveVisibilityRecord": (".visibility", "CurveVisibilityRecord"),
    "QuadricVisibilityError": (".visibility", "QuadricVisibilityError"),
    "canonical_quadric_visibility_json": (
        ".visibility",
        "canonical_quadric_visibility_json",
    ),
    "compute_curve_visibility": (".visibility", "compute_curve_visibility"),
    "compute_quadric_visibility": (".visibility", "compute_quadric_visibility"),
    "DEFAULT_PLANE_PATCH_MARGIN_RATIO": (
        ".plane_patch",
        "DEFAULT_PLANE_PATCH_MARGIN_RATIO",
    ),
    "FittedPlaneDisplayPatch": (".plane_patch", "FittedPlaneDisplayPatch"),
    "PLANE_PATCH_FIT_SCHEMA": (".plane_patch", "PLANE_PATCH_FIT_SCHEMA"),
    "PlanePatchFitError": (".plane_patch", "PlanePatchFitError"),
    "SurfacePlaneExtents": (".plane_patch", "SurfacePlaneExtents"),
    "canonical_fitted_plane_display_patch_json": (
        ".plane_patch",
        "canonical_fitted_plane_display_patch_json",
    ),
    "finite_surface_support_interval": (
        ".plane_patch",
        "finite_surface_support_interval",
    ),
    "fit_plane_display_patch": (".plane_patch", "fit_plane_display_patch"),
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
    "QUADRIC_MANIM_LIMITS": (".manim", "QUADRIC_MANIM_LIMITS"),
    "QuadricManimCapacityError": (".manim", "QuadricManimCapacityError"),
    "QuadricManimError": (".manim", "QuadricManimError"),
    "QuadricManimLimits": (".manim", "QuadricManimLimits"),
    "QuadricManimStyle": (".manim", "QuadricManimStyle"),
    "QuadricOcclusion3D": (".manim", "QuadricOcclusion3D"),
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
