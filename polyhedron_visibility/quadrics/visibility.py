"""Renderer-neutral visibility for finite analytic curves and quadrics.

Critical parameters are constructed analytically by :mod:`.critical`.  The
open cells between those parameters are classified at one midpoint using the
finite solid contracts.  This keeps topology exact while ensuring that an
infinite cylinder or cone support never hides geometry outside its authored
axial range.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from typing import Sequence

from ..geometry import GeometryQuantity
from ..parallel_solver import ParallelView
from ..topology import (
    ParameterInterval,
    assert_exact_partition,
    partition_parameter_domain,
)
from ..visibility import (
    OcclusionInterval,
    VisibilityBoundaryMode,
    VisibilityKind,
    VisibilitySpan,
    partition_visibility,
)
from .critical import (
    AnalyticCurve3D,
    ContextInput,
    CriticalEvent,
    QuadricSurfaceSpec,
    _resolved_context,
    compute_curve_critical_events,
)
from .contract import ConeSpec, CylinderSpec, SphereSpec
from .curves import EllipseArcCurve, ParametricConicBranch, SegmentCurve


QUADRIC_VISIBILITY_RECORD_SCHEMA = "manim-quadric-curve-visibility/v1"
QUADRIC_VISIBILITY_FRAME_SCHEMA = "manim-quadric-visibility-frame/v1"


class QuadricVisibilityError(ValueError):
    """A finite-quadric visibility frame cannot be formed unambiguously."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuadricVisibilityError(f"{label} must be a non-empty string")
    return value.strip()


def _span_dict(span: VisibilitySpan[str]) -> dict[str, object]:
    return {
        "interval": [span.interval.start, span.interval.end],
        "kind": span.kind.value,
        "occluders": list(span.occluders),
    }


@dataclass(frozen=True, slots=True)
class CurveVisibilityRecord:
    """An exact, deterministic visibility partition for one authored curve."""

    curve_id: str
    domain: ParameterInterval
    critical_events: tuple[CriticalEvent, ...]
    spans: tuple[VisibilitySpan[str], ...]
    parameter_tolerance: float
    schema: str = QUADRIC_VISIBILITY_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != QUADRIC_VISIBILITY_RECORD_SCHEMA:
            raise QuadricVisibilityError("invalid curve-visibility schema")
        curve_id = _identity(self.curve_id, "curve_id")
        if not isinstance(self.domain, ParameterInterval):
            raise TypeError("domain must be a ParameterInterval")
        if not isinstance(self.critical_events, tuple) or not all(
            isinstance(item, CriticalEvent) for item in self.critical_events
        ):
            raise TypeError("critical_events must be a tuple of CriticalEvent objects")
        if not isinstance(self.spans, tuple) or not all(
            isinstance(item, VisibilitySpan) for item in self.spans
        ):
            raise TypeError("spans must be a tuple of VisibilitySpan objects")
        tolerance = float(self.parameter_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise QuadricVisibilityError(
                "parameter_tolerance must be finite and non-negative"
            )

        previous = float("-inf")
        for event in self.critical_events:
            if not self.domain.contains(event.parameter, tolerance=tolerance):
                raise QuadricVisibilityError("critical event lies outside curve domain")
            if event.parameter < previous:
                raise QuadricVisibilityError("critical events must be ordered")
            previous = event.parameter
        for span in self.spans:
            if not all(isinstance(owner, str) and owner for owner in span.occluders):
                raise QuadricVisibilityError("span occluders must be non-empty strings")
            if tuple(sorted(span.occluders)) != span.occluders:
                raise QuadricVisibilityError("span occluders must be sorted")
        assert_exact_partition(
            self.domain,
            (span.interval for span in self.spans),
            tolerance=tolerance,
        )
        object.__setattr__(self, "curve_id", curve_id)
        object.__setattr__(self, "parameter_tolerance", tolerance)

    @property
    def visible_intervals(self) -> tuple[ParameterInterval, ...]:
        return tuple(
            span.interval for span in self.spans if span.kind is VisibilityKind.VISIBLE
        )

    @property
    def hidden_intervals(self) -> tuple[ParameterInterval, ...]:
        return tuple(
            span.interval for span in self.spans if span.kind is VisibilityKind.HIDDEN
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "curveId": self.curve_id,
            "domain": [self.domain.start, self.domain.end],
            "parameterTolerance": self.parameter_tolerance,
            "criticalEvents": [event.to_dict() for event in self.critical_events],
            "spans": [_span_dict(span) for span in self.spans],
        }


@dataclass(frozen=True, slots=True)
class CurveVisibilityFrame:
    """Visibility records for one parallel view and one finite-solid set."""

    projection_matrix: tuple[tuple[float, float, float], ...]
    view_direction: tuple[float, float, float]
    surface_ids: tuple[str, ...]
    records: tuple[CurveVisibilityRecord, ...]
    schema: str = QUADRIC_VISIBILITY_FRAME_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != QUADRIC_VISIBILITY_FRAME_SCHEMA:
            raise QuadricVisibilityError("invalid quadric-visibility frame schema")
        view = ParallelView(self.projection_matrix, self.view_direction)
        surfaces = tuple(_identity(item, "surface_id") for item in self.surface_ids)
        if surfaces != tuple(sorted(surfaces)) or len(set(surfaces)) != len(surfaces):
            raise QuadricVisibilityError("surface_ids must be unique and sorted")
        if not isinstance(self.records, tuple) or not all(
            isinstance(item, CurveVisibilityRecord) for item in self.records
        ):
            raise TypeError("records must be a tuple of CurveVisibilityRecord objects")
        curve_ids = tuple(item.curve_id for item in self.records)
        if curve_ids != tuple(sorted(curve_ids)) or len(set(curve_ids)) != len(curve_ids):
            raise QuadricVisibilityError("records must have unique sorted curve identities")
        unknown = sorted(
            {
                owner
                for record in self.records
                for span in record.spans
                for owner in span.occluders
                if owner not in surfaces
            }
        )
        if unknown:
            raise QuadricVisibilityError(
                "visibility spans name unknown surfaces: " + ", ".join(unknown)
            )
        object.__setattr__(self, "projection_matrix", view.projection_matrix)
        object.__setattr__(self, "view_direction", view.view_direction)
        object.__setattr__(self, "surface_ids", surfaces)

    @property
    def record_map(self) -> dict[str, CurveVisibilityRecord]:
        return {record.curve_id: record for record in self.records}

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "projectionMatrix": [list(row) for row in self.projection_matrix],
            "viewDirection": list(self.view_direction),
            "surfaceIds": list(self.surface_ids),
            "records": [record.to_dict() for record in self.records],
        }


def _validated_surfaces(
    surfaces: Sequence[QuadricSurfaceSpec],
) -> tuple[QuadricSurfaceSpec, ...]:
    items = tuple(surfaces)
    if not all(isinstance(item, (SphereSpec, CylinderSpec, ConeSpec)) for item in items):
        raise TypeError("surfaces must contain sphere, cylinder, or cone specs")
    identities = tuple(item.surface_id for item in items)
    if len(set(identities)) != len(identities):
        raise QuadricVisibilityError("surface identities must be unique")
    return tuple(sorted(items, key=lambda item: item.surface_id))


def _occludes_midpoint(
    surface: QuadricSurfaceSpec,
    point: tuple[float, float, float],
    view: ParallelView,
    *,
    context: object,
    depth_epsilon: float,
) -> bool:
    hits = surface.ray_hits(
        point,
        view.view_direction,
        context=context,
        include_caps=True,
        forward_only=True,
    )
    return any(
        hit.parameter > depth_epsilon and not hit.tangential
        for hit in hits
    )


def compute_curve_visibility(
    curve: AnalyticCurve3D,
    surfaces: Sequence[QuadricSurfaceSpec],
    view: ParallelView,
    *,
    context: ContextInput = None,
) -> CurveVisibilityRecord:
    """Partition one curve by analytic events and classify each open cell.

    A zero-depth hit is the curve point itself and is ignored.  A purely
    tangential hit does not put opaque material in front of the point and is
    also ignored.  Another positive-depth hit from that same finite surface
    remains active, which is exactly the front/back test needed for a curve on
    a sphere, cylinder, or cone.
    """

    if not isinstance(curve, (SegmentCurve, EllipseArcCurve, ParametricConicBranch)):
        raise TypeError("curve must be a supported analytic 3D curve")
    if not isinstance(view, ParallelView):
        raise TypeError("view must be a ParallelView")
    surface_items = _validated_surfaces(surfaces)
    resolved = _resolved_context(curve, surface_items, context)
    parameter_epsilon = resolved.epsilon(GeometryQuantity.PARAMETER)
    depth_epsilon = resolved.epsilon(GeometryQuantity.DEPTH)
    events = compute_curve_critical_events(
        curve,
        surface_items,
        view,
        context=resolved,
    )
    cells = partition_parameter_domain(
        curve.domain,
        (event.parameter for event in events),
        tolerance=parameter_epsilon,
    )
    hidden: list[OcclusionInterval[str]] = []
    for cell in cells:
        point = curve.point(cell.midpoint)
        for surface in surface_items:
            if _occludes_midpoint(
                surface,
                point,
                view,
                context=resolved,
                depth_epsilon=depth_epsilon,
            ):
                hidden.append(OcclusionInterval(cell, surface.surface_id))

    spans = partition_visibility(
        curve.domain,
        hidden,
        context=resolved,
        parameter_tolerance=parameter_epsilon,
        occluder_key=lambda owner: owner,
        boundary_mode=VisibilityBoundaryMode.EXACT,
    )
    assert_exact_partition(
        curve.domain,
        (span.interval for span in spans),
        tolerance=parameter_epsilon,
    )
    return CurveVisibilityRecord(
        curve_id=curve.curve_id,
        domain=curve.domain,
        critical_events=events,
        spans=spans,
        parameter_tolerance=parameter_epsilon,
    )


def compute_quadric_visibility(
    curves: Sequence[AnalyticCurve3D],
    surfaces: Sequence[QuadricSurfaceSpec],
    view: ParallelView,
    *,
    context: ContextInput = None,
) -> CurveVisibilityFrame:
    """Compute deterministic independent curve visibility for one frame.

    Multiple finite occluders participate in every curve query.  This API does
    not order transparent surface patches against one another; that separate
    surface-to-surface painter problem is intentionally outside this kernel.
    """

    if not isinstance(view, ParallelView):
        raise TypeError("view must be a ParallelView")
    curve_items = tuple(curves)
    if not all(
        isinstance(item, (SegmentCurve, EllipseArcCurve, ParametricConicBranch))
        for item in curve_items
    ):
        raise TypeError("curves must contain supported analytic 3D curves")
    curve_ids = tuple(item.curve_id for item in curve_items)
    if len(set(curve_ids)) != len(curve_ids):
        raise QuadricVisibilityError("curve identities must be unique")
    curve_items = tuple(sorted(curve_items, key=lambda item: item.curve_id))
    surface_items = _validated_surfaces(surfaces)
    records = tuple(
        compute_curve_visibility(item, surface_items, view, context=context)
        for item in curve_items
    )
    return CurveVisibilityFrame(
        projection_matrix=view.projection_matrix,
        view_direction=view.view_direction,
        surface_ids=tuple(item.surface_id for item in surface_items),
        records=records,
    )


def canonical_quadric_visibility_json(frame: CurveVisibilityFrame) -> str:
    """Serialize one visibility frame with a deterministic byte ordering."""

    if not isinstance(frame, CurveVisibilityFrame):
        raise TypeError("frame must be a CurveVisibilityFrame")
    return json.dumps(
        frame.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "QUADRIC_VISIBILITY_FRAME_SCHEMA",
    "QUADRIC_VISIBILITY_RECORD_SCHEMA",
    "CurveVisibilityFrame",
    "CurveVisibilityRecord",
    "QuadricVisibilityError",
    "canonical_quadric_visibility_json",
    "compute_curve_visibility",
    "compute_quadric_visibility",
]
