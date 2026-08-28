"""Display-only opaque projection proxies for finite convex quadrics.

The objects returned here are two-dimensional paint geometry.  They are an
adaptive approximation of the projected outline and deliberately contain no
world-space surface, ray, depth, or occluder data.  Visibility solvers must
continue to use the exact quadric contracts rather than feeding this proxy
back into geometric truth.

For a screen-space support direction ``d``, the corresponding world-space
linear functional is ``projection[:2].T @ d``.  Sphere, cylinder, and
single-nappe cone/frustum support points are then evaluated analytically.  An
adaptive normal-angle subdivision turns those support points into one closed,
counter-clockwise boundary suitable for a single opaque Manim fill object.
Manim itself is not imported by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import acos, ceil, cos, isfinite, sin, tau
from typing import Sequence

import numpy as np

from ..parallel_solver import ParallelView, SolverError
from .contract import (
    CircularTrimRimSpec,
    ConeSpec,
    CylinderSpec,
    PlanarCapSpec,
    SphereSpec,
)


OPAQUE_PROJECTION_PROXY_SCHEMA = "manim-quadric-opaque-projection-proxy/v1"
CONE_PROJECTION_LAYERS_SCHEMA = "manim-cone-projection-layers/v1"
_INITIAL_SEGMENTS = 8
_PROBE_FRACTIONS = (0.25, 0.5, 0.75)
_EVENT_SCAN_SEGMENTS = 256


QuadricSurfaceSpec = SphereSpec | CylinderSpec | ConeSpec
ParallelViewInput = ParallelView | Sequence[Sequence[float]]


class ProjectionProxyError(ValueError):
    """A display proxy cannot be generated without guessing."""


class ProjectionSubdivisionError(ProjectionProxyError):
    """The requested screen error needs more than ``max_segments``."""

    def __init__(
        self,
        *,
        max_chord_error: float,
        observed_chord_error: float,
        max_segments: int,
        current_segments: int,
    ) -> None:
        self.max_chord_error = float(max_chord_error)
        self.observed_chord_error = float(observed_chord_error)
        self.max_segments = int(max_segments)
        self.current_segments = int(current_segments)
        super().__init__(
            "adaptive projection outline exceeded max_segments "
            f"({self.max_segments}) before reaching max_chord_error "
            f"({self.max_chord_error:.17g}); observed "
            f"{self.observed_chord_error:.17g} at "
            f"{self.current_segments} segments"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "max_segments_exceeded",
            "maxChordError": self.max_chord_error,
            "observedChordError": self.observed_chord_error,
            "maxSegments": self.max_segments,
            "currentSegments": self.current_segments,
        }


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectionProxyError(f"{label} must be a non-empty string")
    return value.strip()


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ProjectionProxyError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProjectionProxyError(f"{label} must be finite and positive") from exc
    if not isfinite(result) or result <= 0.0:
        raise ProjectionProxyError(f"{label} must be finite and positive")
    return result


def _segment_limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProjectionProxyError("max_segments must be an integer")
    if value < _INITIAL_SEGMENTS:
        raise ProjectionProxyError(
            f"max_segments must be at least {_INITIAL_SEGMENTS}"
        )
    return value


def _point2(value: object, label: str) -> tuple[float, float]:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProjectionProxyError(f"{label} must contain two finite values") from exc
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ProjectionProxyError(f"{label} must contain two finite values")
    return float(result[0]), float(result[1])


def _coerce_view(view: ParallelViewInput) -> ParallelView:
    if isinstance(view, ParallelView):
        return view
    try:
        return ParallelView.from_matrix(view)
    except (SolverError, TypeError, ValueError) as exc:
        raise ProjectionProxyError(f"invalid parallel projection: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ProjectionApproximationMetadata:
    """Screen-space approximation diagnostics, never visibility evidence."""

    max_chord_error: float
    observed_chord_error: float
    max_segments: int
    segment_count: int
    adaptive_interval_count: int
    support_evaluation_count: int
    visibility_authoritative: bool = False

    def __post_init__(self) -> None:
        maximum = _positive(self.max_chord_error, "max_chord_error")
        observed = float(self.observed_chord_error)
        if not isfinite(observed) or observed < 0.0:
            raise ProjectionProxyError(
                "observed_chord_error must be finite and non-negative"
            )
        limit = _segment_limit(self.max_segments)
        for label in (
            "segment_count",
            "adaptive_interval_count",
            "support_evaluation_count",
        ):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ProjectionProxyError(f"{label} must be a positive integer")
        if self.segment_count > self.max_segments:
            raise ProjectionProxyError("segment_count exceeds max_segments")
        if observed > maximum * (1.0 + 64.0 * np.finfo(float).eps):
            raise ProjectionProxyError(
                "observed_chord_error exceeds requested max_chord_error"
            )
        if self.visibility_authoritative is not False:
            raise ProjectionProxyError(
                "projection proxies cannot be visibility-authoritative"
            )
        object.__setattr__(self, "max_chord_error", maximum)
        object.__setattr__(self, "observed_chord_error", observed)
        object.__setattr__(self, "max_segments", limit)

    def to_dict(self) -> dict[str, object]:
        return {
            "maxChordError": self.max_chord_error,
            "observedChordError": self.observed_chord_error,
            "maxSegments": self.max_segments,
            "segmentCount": self.segment_count,
            "adaptiveIntervalCount": self.adaptive_interval_count,
            "supportEvaluationCount": self.support_evaluation_count,
            "visibilityAuthoritative": self.visibility_authoritative,
        }


@dataclass(frozen=True, slots=True)
class OpaqueProjectionProxy:
    """One deterministic closed 2D boundary for an opaque display fill."""

    patch_id: str
    surface_id: str
    boundary_points: tuple[tuple[float, float], ...]
    metadata: ProjectionApproximationMetadata
    schema: str = OPAQUE_PROJECTION_PROXY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != OPAQUE_PROJECTION_PROXY_SCHEMA:
            raise ProjectionProxyError("invalid opaque-projection proxy schema")
        patch_id = _identity(self.patch_id, "patch_id")
        surface_id = _identity(self.surface_id, "surface_id")
        points = tuple(
            _point2(point, "boundary point") for point in self.boundary_points
        )
        if len(points) < 4:
            raise ProjectionProxyError(
                "a closed projection boundary requires at least three vertices"
            )
        if points[0] != points[-1]:
            raise ProjectionProxyError(
                "projection boundary must repeat its first point exactly"
            )
        if len(set(points[:-1])) < 3:
            raise ProjectionProxyError(
                "projection boundary requires at least three distinct vertices"
            )
        if not isinstance(self.metadata, ProjectionApproximationMetadata):
            raise TypeError("metadata must be ProjectionApproximationMetadata")
        if self.metadata.segment_count != len(points) - 1:
            raise ProjectionProxyError(
                "metadata segment_count disagrees with projection boundary"
            )
        anchor = np.asarray(points[0], dtype=float)
        local_points = tuple(
            np.asarray(point, dtype=float) - anchor for point in points
        )
        area_twice = sum(
            left[0] * right[1] - left[1] * right[0]
            for left, right in zip(local_points, local_points[1:])
        )
        if not isfinite(area_twice) or area_twice <= 0.0:
            raise ProjectionProxyError(
                "projection boundary must be non-degenerate and counter-clockwise"
            )
        object.__setattr__(self, "patch_id", patch_id)
        object.__setattr__(self, "surface_id", surface_id)
        object.__setattr__(self, "boundary_points", points)

    @property
    def vertices(self) -> tuple[tuple[float, float], ...]:
        """Boundary vertices without the repeated closure point."""

        return self.boundary_points[:-1]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "patchId": self.patch_id,
            "surfaceId": self.surface_id,
            "boundaryPoints": [list(point) for point in self.boundary_points],
            "metadata": self.metadata.to_dict(),
        }


ProjectionPath = tuple[tuple[float, float], ...]


def _projection_path(
    value: Sequence[Sequence[float]],
    label: str,
) -> ProjectionPath:
    points = tuple(_point2(point, label) for point in value)
    if len(points) < 3 or len(set(points)) < 3:
        raise ProjectionProxyError(f"{label} requires three distinct points")
    return points


@dataclass(frozen=True, slots=True)
class ConeProjectionSheet:
    """Component masks for one far or near cone projection sheet."""

    lateral_paths: tuple[ProjectionPath, ...]
    cap_paths: tuple[ProjectionPath, ...] = ()

    def __post_init__(self) -> None:
        lateral = tuple(
            _projection_path(path, "cone lateral path")
            for path in self.lateral_paths
        )
        caps = tuple(
            _projection_path(path, "cone cap path") for path in self.cap_paths
        )
        if not lateral:
            raise ProjectionProxyError(
                "a cone projection sheet requires a lateral footprint"
            )
        object.__setattr__(self, "lateral_paths", lateral)
        object.__setattr__(self, "cap_paths", caps)


@dataclass(frozen=True, slots=True)
class ConeProjectionLayers:
    """Renderer-neutral cap/lateral masks for one finite single-nappe cone."""

    surface_id: str
    proxy: OpaqueProjectionProxy
    back: ConeProjectionSheet
    front: ConeProjectionSheet
    opaque_lateral_paths: tuple[ProjectionPath, ...]
    opaque_cap_paths: tuple[ProjectionPath, ...]
    terminal_front_facing: bool | None
    terminal_front_facing_by_id: tuple[tuple[str, bool | None], ...] = ()
    schema: str = CONE_PROJECTION_LAYERS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CONE_PROJECTION_LAYERS_SCHEMA:
            raise ProjectionProxyError("invalid cone projection-layers schema")
        surface_id = _identity(self.surface_id, "surface_id")
        if self.proxy.surface_id != surface_id:
            raise ProjectionProxyError(
                "cone projection layers and proxy must share a surface identity"
            )
        if not isinstance(self.back, ConeProjectionSheet) or not isinstance(
            self.front, ConeProjectionSheet
        ):
            raise TypeError("back and front must be ConeProjectionSheet values")
        opaque_lateral = tuple(
            _projection_path(path, "opaque cone lateral path")
            for path in self.opaque_lateral_paths
        )
        opaque_caps = tuple(
            _projection_path(path, "opaque cone cap path")
            for path in self.opaque_cap_paths
        )
        if not opaque_lateral:
            raise ProjectionProxyError(
                "opaque cone projection requires a lateral footprint"
            )
        if self.terminal_front_facing not in {None, True, False}:
            raise TypeError("terminal_front_facing must be bool or None")
        terminal_facing: list[tuple[str, bool | None]] = []
        seen_terminal_ids: set[str] = set()
        for raw in self.terminal_front_facing_by_id:
            if not isinstance(raw, tuple) or len(raw) != 2:
                raise TypeError(
                    "terminal_front_facing_by_id values must be (id, facing) pairs"
                )
            terminal_id = _identity(raw[0], "terminal identity")
            facing = raw[1]
            if facing is not None and not isinstance(facing, bool):
                raise TypeError("terminal facing values must be bool or None")
            if terminal_id in seen_terminal_ids:
                raise ProjectionProxyError(
                    "terminal_front_facing_by_id identities must be unique"
                )
            seen_terminal_ids.add(terminal_id)
            terminal_facing.append((terminal_id, facing))
        terminal_facing.sort(key=lambda item: item[0])
        if len(terminal_facing) == 1:
            mapped_facing = terminal_facing[0][1]
            if self.terminal_front_facing != mapped_facing:
                raise ProjectionProxyError(
                    "legacy terminal_front_facing disagrees with its identity map"
                )
        elif len(terminal_facing) > 1 and self.terminal_front_facing is not None:
            raise ProjectionProxyError(
                "legacy terminal_front_facing is only defined for one terminal"
            )
        object.__setattr__(self, "surface_id", surface_id)
        object.__setattr__(self, "opaque_lateral_paths", opaque_lateral)
        object.__setattr__(self, "opaque_cap_paths", opaque_caps)
        object.__setattr__(
            self,
            "terminal_front_facing_by_id",
            tuple(terminal_facing),
        )


def canonical_opaque_projection_proxy_json(proxy: OpaqueProjectionProxy) -> str:
    if not isinstance(proxy, OpaqueProjectionProxy):
        raise TypeError("proxy must be an OpaqueProjectionProxy")
    return json.dumps(
        proxy.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _support_sphere(surface: SphereSpec, covector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(covector))
    if not isfinite(length) or length <= 0.0:
        raise ProjectionProxyError("screen direction has no world-space pullback")
    return (
        np.asarray(surface.center, dtype=float)
        + surface.radius * covector / length
    )


def _radial_support(
    covector: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    radius: float,
) -> np.ndarray:
    first = float(np.dot(covector, x_axis))
    second = float(np.dot(covector, y_axis))
    length = float(np.hypot(first, second))
    if length == 0.0:
        return np.zeros(3, dtype=float)
    return radius * (first * x_axis + second * y_axis) / length


def _support_cylinder(
    surface: CylinderSpec,
    covector: np.ndarray,
    feature: int | None = None,
) -> np.ndarray:
    frame = surface.frame
    axis = np.asarray(frame.z_axis, dtype=float)
    x_axis = np.asarray(frame.x_axis, dtype=float)
    y_axis = np.asarray(frame.y_axis, dtype=float)
    lower, upper = surface.axial_range
    if feature is None:
        feature = 1 if float(np.dot(covector, axis)) > 0.0 else 0
    axial = (lower, upper)[feature]
    return (
        np.asarray(surface.origin, dtype=float)
        + axial * axis
        + _radial_support(covector, x_axis, y_axis, surface.radius)
    )


def _support_cone(
    surface: ConeSpec,
    covector: np.ndarray,
    feature: int | None = None,
) -> np.ndarray:
    lower, upper = surface.axial_range
    if lower < 0.0 < upper:
        raise ProjectionProxyError(
            "a cone axial_range crossing zero contains two nappes; split it into "
            "two single-nappe surfaces before building projection proxies"
        )
    frame = surface.frame
    apex = np.asarray(surface.apex, dtype=float)
    axis = np.asarray(frame.z_axis, dtype=float)
    x_axis = np.asarray(frame.x_axis, dtype=float)
    y_axis = np.asarray(frame.y_axis, dtype=float)
    candidates: list[tuple[float, np.ndarray]] = []
    for axial in (lower, upper):
        radius = abs(axial) * surface.slope
        point = (
            apex
            + axial * axis
            + _radial_support(covector, x_axis, y_axis, radius)
        )
        candidates.append((float(np.dot(covector, point)), point))

    # Keep the lower axial endpoint on a numerical tie.  The adjacent support
    # directions approach both endpoints and the adaptive outline inserts the
    # straight projected support face between them.
    first_value, first_point = candidates[0]
    second_value, second_point = candidates[1]
    scale = max(abs(first_value), abs(second_value), np.finfo(float).tiny)
    tie = 64.0 * np.finfo(float).eps * scale
    if feature is not None:
        return (first_point, second_point)[feature]
    return second_point if second_value > first_value + tie else first_point


def _support_world(
    surface: QuadricSurfaceSpec,
    covector: np.ndarray,
    feature: int | None = None,
) -> np.ndarray:
    if isinstance(surface, SphereSpec):
        if feature is not None:
            raise ProjectionProxyError("a sphere has no discrete support feature")
        return _support_sphere(surface, covector)
    if isinstance(surface, CylinderSpec):
        return _support_cylinder(surface, covector, feature)
    if isinstance(surface, ConeSpec):
        return _support_cone(surface, covector, feature)
    raise TypeError("surface must be SphereSpec, CylinderSpec, or ConeSpec")


def _feature_score_difference(
    surface: CylinderSpec | ConeSpec,
    covector: np.ndarray,
) -> float:
    """Return upper-end support score minus lower-end support score."""

    frame = surface.frame
    axis = np.asarray(frame.z_axis, dtype=float)
    lower, upper = surface.axial_range
    axial = (upper - lower) * float(np.dot(covector, axis))
    if isinstance(surface, CylinderSpec):
        return axial
    x_axis = np.asarray(frame.x_axis, dtype=float)
    y_axis = np.asarray(frame.y_axis, dtype=float)
    radial_norm = float(
        np.hypot(np.dot(covector, x_axis), np.dot(covector, y_axis))
    )
    radial = (abs(upper) - abs(lower)) * surface.slope * radial_norm
    return axial + radial


def _feature_events(
    surface: QuadricSurfaceSpec,
    screen_matrix: np.ndarray,
) -> tuple[float, ...]:
    """Find normal angles where two axial support faces exchange ownership.

    Each sign-changing event becomes an exact straight boundary edge between
    the two endpoint support points.  This avoids asking angular subdivision
    to approximate a discontinuous choice of support point.
    """

    if isinstance(surface, SphereSpec):
        return ()

    def score(angle: float) -> float:
        direction = np.asarray((cos(angle), sin(angle)), dtype=float)
        return _feature_score_difference(surface, screen_matrix.T @ direction)

    step = tau / _EVENT_SCAN_SEGMENTS
    samples = [score(index * step) for index in range(_EVENT_SCAN_SEGMENTS)]
    score_scale = max((abs(value) for value in samples), default=0.0)
    numerical = 256.0 * np.finfo(float).eps * max(
        score_scale, np.finfo(float).tiny
    )
    if score_scale <= numerical:
        # The cylinder axis projects to a point.  Its two end disks have the
        # same screen projection, so no visible support-face edge is needed.
        return ()

    def sign(value: float) -> int:
        if value > numerical:
            return 1
        if value < -numerical:
            return -1
        return 0

    roots: list[float] = []
    for index in range(_EVENT_SCAN_SEGMENTS):
        start = index * step
        end = (index + 1) * step
        start_value = samples[index]
        end_value = samples[(index + 1) % _EVENT_SCAN_SEGMENTS]
        start_sign = sign(start_value)
        end_sign = sign(end_value)
        if start_sign == 0:
            before = sign(score(start - 0.25 * step))
            after = sign(score(start + 0.25 * step))
            if before * after < 0:
                roots.append(start % tau)
            continue
        if end_sign == 0 or start_sign == end_sign:
            continue
        left = start
        right = end
        left_sign = start_sign
        for _ in range(80):
            midpoint = left + 0.5 * (right - left)
            if midpoint == left or midpoint == right:
                break
            middle_sign = sign(score(midpoint))
            if middle_sign == 0:
                left = right = midpoint
                break
            if middle_sign == left_sign:
                left = midpoint
            else:
                right = midpoint
        roots.append((left + 0.5 * (right - left)) % tau)

    roots.sort()
    clustered: list[float] = []
    angular_epsilon = 1024.0 * np.finfo(float).eps
    for root in roots:
        if not clustered or root - clustered[-1] > angular_epsilon:
            clustered.append(root)
    if (
        len(clustered) > 1
        and clustered[0] + tau - clustered[-1] <= angular_epsilon
    ):
        clustered[0] = 0.0
        clustered.pop()
    return tuple(clustered)


def _point_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    displacement = end - start
    squared_length = float(np.dot(displacement, displacement))
    if squared_length == 0.0:
        return float(np.linalg.norm(point - start))
    ratio = float(np.dot(point - start, displacement) / squared_length)
    ratio = min(1.0, max(0.0, ratio))
    return float(np.linalg.norm(point - (start + ratio * displacement)))


def _screen_ulp_floor(points: Sequence[np.ndarray]) -> float:
    """Conservative screen-distance uncertainty for stored float coordinates."""

    return 4.0 * max(
        (
            abs(float(np.spacing(value)))
            for point in points
            for value in np.asarray(point, dtype=float)
        ),
        default=0.0,
    )


def _dedupe_consecutive(
    points: Sequence[np.ndarray],
    *,
    max_chord_error: float,
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    if not points:
        return (), ()

    anchor = np.asarray(points[0], dtype=float)

    def duplicate_tolerance(left: np.ndarray, right: np.ndarray) -> float:
        # Translation must not inflate the ordinary round-off allowance.  The
        # local term follows the outline's extent, while the ULP term accounts
        # for the actual representability of large absolute screen positions.
        local_scale = max(
            float(np.linalg.norm(left - anchor)),
            float(np.linalg.norm(right - anchor)),
            max_chord_error,
            np.finfo(float).tiny,
        )
        local_roundoff = 32.0 * np.finfo(float).eps * local_scale
        ulp_roundoff = 2.0 * max(
            *(abs(float(np.spacing(value))) for value in left),
            *(abs(float(np.spacing(value))) for value in right),
        )
        # Never let a large world-space translation turn the requested visual
        # tolerance itself into a duplicate.  If the remaining floating-point
        # resolution is too coarse, the post-deduplication certification below
        # fails closed instead of silently weakening max_chord_error.
        return min(
            max(local_roundoff, ulp_roundoff),
            0.125 * max_chord_error,
        )

    result = [anchor]
    source_to_result = [0]
    for point in points[1:]:
        value = np.asarray(point, dtype=float)
        if float(np.linalg.norm(value - result[-1])) > duplicate_tolerance(
            result[-1], value
        ):
            result.append(value)
        source_to_result.append(len(result) - 1)
    if (
        len(result) > 1
        and float(np.linalg.norm(result[-1] - result[0]))
        <= duplicate_tolerance(result[-1], result[0])
    ):
        removed_index = len(result) - 1
        result.pop()
        source_to_result = [
            0 if index == removed_index else index for index in source_to_result
        ]
    if len(result) < 3:
        raise ProjectionProxyError("projected quadric outline is degenerate")

    return tuple(result), tuple(source_to_result)


def _canonical_boundary(
    points: Sequence[np.ndarray],
) -> tuple[tuple[float, float], ...]:
    if len(points) < 3:
        raise ProjectionProxyError("projected quadric outline is degenerate")

    anchor = np.asarray(points[0], dtype=float)
    local_points = [np.asarray(point, dtype=float) - anchor for point in points]
    area_twice = sum(
        float(left[0] * right[1] - left[1] * right[0])
        for left, right in zip(
            local_points,
            (*local_points[1:], local_points[0]),
        )
    )
    if area_twice < 0.0:
        first = points[0]
        points = (first, *reversed(points[1:]))
    elif area_twice == 0.0:
        raise ProjectionProxyError("projected quadric outline has zero area")
    canonical = tuple((float(point[0]), float(point[1])) for point in points)
    return (*canonical, canonical[0])


def build_opaque_projection_proxy(
    surface: QuadricSurfaceSpec,
    view: ParallelViewInput,
    *,
    patch_id: str | None = None,
    max_chord_error: float = 1.0e-3,
    max_segments: int = 4096,
) -> OpaqueProjectionProxy:
    """Build one display-only closed silhouette proxy under a parallel view.

    ``max_chord_error`` is measured in the two-dimensional coordinate system
    produced by the first two projection rows.  If adaptive subdivision would
    exceed ``max_segments``, no partial proxy is returned.
    """

    if not isinstance(surface, (SphereSpec, CylinderSpec, ConeSpec)):
        raise TypeError("surface must be SphereSpec, CylinderSpec, or ConeSpec")
    parallel_view = _coerce_view(view)
    tolerance = _positive(max_chord_error, "max_chord_error")
    segment_limit = _segment_limit(max_segments)
    identity = (
        f"{surface.surface_id}:opaque-projection"
        if patch_id is None
        else _identity(patch_id, "patch_id")
    )
    if isinstance(surface, ConeSpec):
        lower, upper = surface.axial_range
        if lower < 0.0 < upper:
            raise ProjectionProxyError(
                "a cone axial_range crossing zero contains two nappes; split it "
                "before building a convex projection proxy"
            )

    screen_matrix = parallel_view.matrix[:2]
    if screen_matrix.shape != (2, 3) or int(np.linalg.matrix_rank(screen_matrix)) != 2:
        raise ProjectionProxyError("parallel projection screen axes are singular")

    support_cache: dict[tuple[float, int | None], np.ndarray] = {}

    def support(angle: float, feature: int | None = None) -> np.ndarray:
        normalized = float(angle % tau)
        if abs(normalized) <= 8.0 * np.finfo(float).eps or abs(tau - normalized) <= (
            8.0 * np.finfo(float).eps
        ):
            normalized = 0.0
        key = (normalized, feature)
        cached = support_cache.get(key)
        if cached is not None:
            return cached
        direction = np.asarray((cos(normalized), sin(normalized)), dtype=float)
        covector = screen_matrix.T @ direction
        world = _support_world(surface, covector, feature)
        projected = np.asarray(screen_matrix @ world, dtype=float)
        if projected.shape != (2,) or not np.all(np.isfinite(projected)):
            raise ProjectionProxyError("quadric support projection is non-finite")
        support_cache[key] = projected
        return projected

    event_angles = _feature_events(surface, screen_matrix)
    breakpoints = [
        index * tau / _INITIAL_SEGMENTS
        for index in range(_INITIAL_SEGMENTS + 1)
    ]
    breakpoints.extend(event_angles)
    breakpoints.sort()
    angular_epsilon = 1024.0 * np.finfo(float).eps
    unique_breakpoints = [0.0]
    for value in breakpoints[1:]:
        normalized = min(tau, max(0.0, float(value)))
        if normalized - unique_breakpoints[-1] > angular_epsilon:
            unique_breakpoints.append(normalized)
    if tau - unique_breakpoints[-1] <= angular_epsilon:
        unique_breakpoints[-1] = tau
    else:
        unique_breakpoints.append(tau)

    def interval_feature(start: float, end: float) -> int | None:
        if isinstance(surface, SphereSpec):
            return None
        midpoint = start + 0.5 * (end - start)
        direction = np.asarray((cos(midpoint), sin(midpoint)), dtype=float)
        difference = _feature_score_difference(
            surface,
            screen_matrix.T @ direction,
        )
        return 1 if difference > 0.0 else 0

    intervals: list[tuple[float, float, int | None]] = [
        (start, end, interval_feature(start, end))
        for start, end in zip(unique_breakpoints, unique_breakpoints[1:])
        if end > start
    ]
    feature_edges = sum(
        left[2] != right[2]
        for left, right in zip(intervals, (*intervals[1:], intervals[0]))
    )
    while True:
        split_indices: list[int] = []
        pass_error = 0.0
        for index, (start, end, feature) in enumerate(intervals):
            start_point = support(start, feature)
            end_point = support(end, feature)
            interval_error = 0.0
            for fraction in _PROBE_FRACTIONS:
                probe = start + fraction * (end - start)
                interval_error = max(
                    interval_error,
                    _point_segment_distance(
                        support(probe, feature), start_point, end_point
                    ),
                )
            pass_error = max(pass_error, interval_error)
            if interval_error > tolerance:
                split_indices.append(index)
        if not split_indices:
            break
        current_segments = len(intervals) + feature_edges
        if current_segments + len(split_indices) > segment_limit:
            raise ProjectionSubdivisionError(
                max_chord_error=tolerance,
                observed_chord_error=pass_error,
                max_segments=segment_limit,
                current_segments=current_segments,
            )
        marked = set(split_indices)
        refined: list[tuple[float, float, int | None]] = []
        for index, (start, end, feature) in enumerate(intervals):
            if index not in marked:
                refined.append((start, end, feature))
                continue
            midpoint = start + 0.5 * (end - start)
            refined.extend(
                ((start, midpoint, feature), (midpoint, end, feature))
            )
        intervals = refined

    sampled_points: list[np.ndarray] = []
    for start, end, feature in intervals:
        sampled_points.append(support(start, feature))
        sampled_points.append(support(end, feature))
    deduped_points, source_to_point = _dedupe_consecutive(
        sampled_points,
        max_chord_error=tolerance,
    )
    precision_floor = _screen_ulp_floor(sampled_points)
    if precision_floor >= tolerance:
        raise ProjectionProxyError(
            "projected quadric outline cannot certify max_chord_error at the "
            "available floating-point screen resolution; requested "
            f"{tolerance:.17g}, resolution floor {precision_floor:.17g}"
        )
    measured_error = 0.0
    certification_fractions = (0.0, *_PROBE_FRACTIONS, 1.0)
    for index, (start, end, feature) in enumerate(intervals):
        chord_start = deduped_points[source_to_point[2 * index]]
        chord_end = deduped_points[source_to_point[2 * index + 1]]
        for fraction in certification_fractions:
            parameter = start + fraction * (end - start)
            measured_error = max(
                measured_error,
                _point_segment_distance(
                    support(parameter, feature),
                    chord_start,
                    chord_end,
                ),
            )
    certified_error = measured_error + precision_floor
    if certified_error > tolerance * (1.0 + 64.0 * np.finfo(float).eps):
        raise ProjectionProxyError(
            "projected quadric outline cannot certify max_chord_error after "
            "floating-point-stable deduplication; requested "
            f"{tolerance:.17g}, observed {certified_error:.17g}"
        )
    boundary = _canonical_boundary(deduped_points)
    if len(boundary) - 1 > segment_limit:
        raise ProjectionSubdivisionError(
            max_chord_error=tolerance,
            observed_chord_error=certified_error,
            max_segments=segment_limit,
            current_segments=len(boundary) - 1,
        )
    metadata = ProjectionApproximationMetadata(
        max_chord_error=tolerance,
        observed_chord_error=certified_error,
        max_segments=segment_limit,
        segment_count=len(boundary) - 1,
        adaptive_interval_count=len(intervals),
        support_evaluation_count=len(support_cache),
        visibility_authoritative=False,
    )
    return OpaqueProjectionProxy(
        patch_id=identity,
        surface_id=surface.surface_id,
        boundary_points=boundary,
        metadata=metadata,
    )


def _signed_path_area(path: Sequence[Sequence[float]]) -> float:
    points = np.asarray(path, dtype=float)
    shifted = np.roll(points, -1, axis=0)
    return 0.5 * float(
        np.sum(points[:, 0] * shifted[:, 1] - points[:, 1] * shifted[:, 0])
    )


ConeTerminalSpec = PlanarCapSpec | CircularTrimRimSpec


@dataclass(frozen=True, slots=True)
class _ProjectedConeTerminal:
    terminal_id: str
    rim: ProjectionPath | None
    front_facing: bool | None


def _cone_terminal_id(terminal: ConeTerminalSpec) -> str:
    return terminal.cap_id if isinstance(terminal, PlanarCapSpec) else terminal.rim_id


def _ellipse_segment_count(
    screen_radius: float,
    *,
    tolerance: float,
    segment_limit: int,
) -> int:
    ratio = min(1.0, max(0.0, tolerance / screen_radius))
    maximum_step = 2.0 * acos(max(-1.0, min(1.0, 1.0 - ratio)))
    if maximum_step <= 0.0:
        observed = screen_radius * (1.0 - cos(0.5 * tau / segment_limit))
        if observed > tolerance:
            raise ProjectionSubdivisionError(
                max_chord_error=tolerance,
                observed_chord_error=observed,
                max_segments=segment_limit,
                current_segments=segment_limit,
            )
        return segment_limit
    count = max(_INITIAL_SEGMENTS, int(ceil(tau / maximum_step)))
    if count > segment_limit:
        observed = screen_radius * (1.0 - cos(0.5 * tau / segment_limit))
        raise ProjectionSubdivisionError(
            max_chord_error=tolerance,
            observed_chord_error=observed,
            max_segments=segment_limit,
            current_segments=segment_limit,
        )
    return count


def _project_cone_terminal(
    terminal: ConeTerminalSpec,
    view: ParallelView,
    *,
    tolerance: float,
    segment_limit: int,
) -> _ProjectedConeTerminal:
    frame = terminal.frame
    screen = np.asarray(view.matrix[:2], dtype=float)
    center = screen @ np.asarray(terminal.center, dtype=float)
    first = terminal.radius * screen @ np.asarray(frame.x_axis, dtype=float)
    second = terminal.radius * screen @ np.asarray(frame.y_axis, dtype=float)
    basis = np.column_stack((first, second))
    singular_values = np.linalg.svd(basis, compute_uv=False)
    screen_radius = float(singular_values[0])
    terminal_id = _cone_terminal_id(terminal)
    if screen_radius <= tolerance:
        return _ProjectedConeTerminal(terminal_id, None, None)

    count = _ellipse_segment_count(
        screen_radius,
        tolerance=tolerance,
        segment_limit=segment_limit,
    )
    rim_points = tuple(
        tuple(
            float(item)
            for item in (
                center
                + cos(index * tau / count) * first
                + sin(index * tau / count) * second
            )
        )
        for index in range(count)
    )
    area = _signed_path_area(rim_points)
    area_floor = max(
        tolerance * tolerance,
        512.0
        * np.finfo(float).eps
        * max(1.0, screen_radius * screen_radius),
    )
    if abs(area) <= area_floor:
        # An edge-on terminal has a real finite boundary segment, but no area
        # to subtract from a fill mask. Semantic boundary compositing owns the
        # segment; component shading must not invent a thin polygon.
        return _ProjectedConeTerminal(terminal_id, None, None)

    rim = rim_points if area > 0.0 else tuple(reversed(rim_points))
    facing = float(
        np.dot(
            np.asarray(terminal.normal, dtype=float),
            np.asarray(view.view_direction, dtype=float),
        )
    )
    return _ProjectedConeTerminal(terminal_id, rim, facing > 0.0)


def build_cone_projection_layers(
    surface: ConeSpec,
    view: ParallelViewInput,
    *,
    max_chord_error: float = 1.0e-3,
    max_segments: int = 4096,
) -> ConeProjectionLayers:
    """Build stable lateral/cap masks for one finite single-nappe cone/frustum.

    An apex-to-rim cone has exactly one non-degenerate terminal circle.  Its
    projected disk replaces one sheet of a closed cone and removes that sheet
    from an open shell.  The opposite sheet remains lateral.  This models the
    lighter one-hit region visible through an open mouth without creating any
    renderer objects.

    A frustum has two terminals with opposite outward normals. Under a
    parallel view, one belongs to the far sheet and the other to the near
    sheet, so each depth sheet needs at most one finite disk subtraction. A
    closed frustum fills those same disks as real caps; an open frustum leaves
    them as openings. Edge-on terminals remain owned by semantic boundary
    strokes and contribute no zero-area fill polygon.
    """

    if not isinstance(surface, ConeSpec):
        raise TypeError("surface must be a ConeSpec")
    if surface.nappe_count != 1:
        raise ProjectionProxyError(
            "cone projection layers require one nappe; expand an open double "
            "cone into its stable render_components first"
        )
    parallel_view = _coerce_view(view)
    tolerance = _positive(max_chord_error, "max_chord_error")
    segment_limit = _segment_limit(max_segments)
    proxy = build_opaque_projection_proxy(
        surface,
        parallel_view,
        max_chord_error=tolerance,
        max_segments=segment_limit,
    )
    terminals = (
        tuple(surface.trim_rims) if surface.is_open_shell else tuple(surface.end_caps)
    )
    if len(terminals) not in {1, 2}:
        raise ProjectionProxyError(
            "component-aware cone shading requires one or two non-degenerate "
            "terminals on a finite single-nappe cone"
        )
    outer = tuple(proxy.vertices)
    projected = tuple(
        _project_cone_terminal(
            terminal,
            parallel_view,
            tolerance=tolerance,
            segment_limit=segment_limit,
        )
        for terminal in terminals
    )
    active = tuple(item for item in projected if item.rim is not None)
    if len(active) == 2 and active[0].front_facing == active[1].front_facing:
        raise ProjectionProxyError(
            "frustum terminals do not separate into opposite projection sheets"
        )

    back_lateral: list[ProjectionPath] = [outer]
    front_lateral: list[ProjectionPath] = [outer]
    back_caps: list[ProjectionPath] = []
    front_caps: list[ProjectionPath] = []
    for terminal in active:
        assert terminal.rim is not None
        hole = tuple(reversed(terminal.rim))
        if terminal.front_facing:
            front_lateral.append(hole)
            if not surface.is_open_shell:
                front_caps.append(terminal.rim)
        else:
            back_lateral.append(hole)
            if not surface.is_open_shell:
                back_caps.append(terminal.rim)

    back = ConeProjectionSheet(tuple(back_lateral), tuple(back_caps))
    front = ConeProjectionSheet(tuple(front_lateral), tuple(front_caps))
    opaque_lateral: tuple[ProjectionPath, ...] = (
        (outer,)
        if surface.is_open_shell
        else (outer, *(tuple(reversed(rim)) for rim in front_caps))
    )
    opaque_caps = () if surface.is_open_shell else tuple(front_caps)
    terminal_facing = tuple(
        sorted(
            ((item.terminal_id, item.front_facing) for item in projected),
            key=lambda item: item[0],
        )
    )
    return ConeProjectionLayers(
        surface_id=surface.surface_id,
        proxy=proxy,
        back=back,
        front=front,
        opaque_lateral_paths=opaque_lateral,
        opaque_cap_paths=opaque_caps,
        terminal_front_facing=(
            projected[0].front_facing if len(projected) == 1 else None
        ),
        terminal_front_facing_by_id=terminal_facing,
    )


__all__ = [
    "CONE_PROJECTION_LAYERS_SCHEMA",
    "ConeProjectionLayers",
    "ConeProjectionSheet",
    "OPAQUE_PROJECTION_PROXY_SCHEMA",
    "OpaqueProjectionProxy",
    "ParallelViewInput",
    "ProjectionApproximationMetadata",
    "ProjectionProxyError",
    "ProjectionSubdivisionError",
    "QuadricSurfaceSpec",
    "build_opaque_projection_proxy",
    "build_cone_projection_layers",
    "canonical_opaque_projection_proxy_json",
]
