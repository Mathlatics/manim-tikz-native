"""Renderer-neutral two-dimensional views of certified Dandelin geometry.

The two diagrams in this module deliberately have different mathematical
meanings:

``DandelinMeridianDiagram2D``
    is the true meridian section through the cone axis and the radial
    projection of the authored section-plane normal.  A Dandelin sphere is
    therefore represented by a genuine great-circle section, and its contacts
    with the section line and the two cone generators can be certified in two
    dimensions.

``DandelinSectionPlaneDiagram2D``
    lives in the authored cutting plane.  It carries the exact conic trace,
    foci, directrices, and sphere-plane contact evidence.  Sphere centres do
    not generally lie in this plane, so this contract intentionally exposes no
    sphere-circle collection.  Drawing circles around the foci in this view
    would invent geometry that the Dandelin construction does not contain.

Both views retain stable semantic identities and use :class:`PlanarFrame3D`
and :class:`PlanarPoint3D` to make every two-dimensional coordinate reversible
to one certified world-space point.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from typing import Sequence

import numpy as np

from ..geometry import GeometryQuantity
from .conics import ConicKind
from .dandelin import DandelinConstruction3D, DandelinSphere3D
from .planar_curves import PlanarFrame3D, PlanarPoint3D
from .sections import QuadricSectionError, compute_quadric_section
from .trace import QuadricSectionTrace


DANDELIN_MERIDIAN_DIAGRAM_2D_SCHEMA = "manim-dandelin-meridian-diagram-2d/v1"
DANDELIN_SECTION_PLANE_DIAGRAM_2D_SCHEMA = (
    "manim-dandelin-section-plane-diagram-2d/v1"
)


class DandelinView2DError(ValueError):
    """A certified Dandelin construction cannot produce one requested view."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DandelinView2DError(f"{label} must be a non-empty string")
    return value.strip()


def _finite(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise DandelinView2DError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DandelinView2DError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise DandelinView2DError(f"{label} must be finite")
    return 0.0 if result == 0.0 else result


def _positive(value: object, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise DandelinView2DError(f"{label} must be positive")
    return result


def _point2(value: Sequence[float], label: str) -> tuple[float, float]:
    try:
        raw = tuple(value)
        if any(isinstance(item, (bool, np.bool_)) for item in raw):
            raise TypeError
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DandelinView2DError(
            f"{label} must contain two finite coordinates"
        ) from exc
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise DandelinView2DError(
            f"{label} must contain two finite coordinates"
        )
    return tuple(  # type: ignore[return-value]
        0.0 if item == 0.0 else float(item) for item in result
    )


def _point3(value: Sequence[float], label: str) -> tuple[float, float, float]:
    try:
        raw = tuple(value)
        if any(isinstance(item, (bool, np.bool_)) for item in raw):
            raise TypeError
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DandelinView2DError(
            f"{label} must contain three finite coordinates"
        ) from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise DandelinView2DError(
            f"{label} must contain three finite coordinates"
        )
    return tuple(  # type: ignore[return-value]
        0.0 if item == 0.0 else float(item) for item in result
    )


def _unit2(value: Sequence[float], label: str) -> tuple[float, float]:
    direction = np.asarray(_point2(value, label), dtype=float)
    length = float(np.linalg.norm(direction))
    if not isfinite(length) or length <= 0.0:
        raise DandelinView2DError(f"{label} must be non-zero")
    direction /= length
    return _point2(direction, label)


def _canonical_direction2(value: Sequence[float], label: str) -> tuple[float, float]:
    direction = np.asarray(_unit2(value, label), dtype=float)
    index = int(np.argmax(np.abs(direction)))
    if direction[index] < 0.0:
        direction = -direction
    return _point2(direction, label)


def _finite_residual(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise DandelinView2DError(f"{label} must be non-negative")
    return result


def _sorted_unique_identities(values: Sequence[str], label: str) -> None:
    identities = tuple(values)
    if len(set(identities)) != len(identities):
        raise DandelinView2DError(f"{label} identities must be unique")
    if identities != tuple(sorted(identities)):
        raise DandelinView2DError(f"{label} must use canonical identity order")


def _same_residual(actual: float, recorded: float) -> bool:
    scale = max(1.0, abs(actual), abs(recorded))
    return abs(actual - recorded) <= 64.0 * np.finfo(float).eps * scale


def _normalize_canonical_numbers(value: object) -> object:
    """Remove signed zero before deterministic JSON encoding."""

    if isinstance(value, float):
        return 0.0 if value == 0.0 else value
    if isinstance(value, list):
        return [_normalize_canonical_numbers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_canonical_numbers(item) for item in value)
    if isinstance(value, dict):
        return {
            key: _normalize_canonical_numbers(item)
            for key, item in value.items()
        }
    return value


def _world_embedding(
    frame: PlanarFrame3D,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    origin = np.asarray(frame.point, dtype=float)
    first = np.asarray(frame.u_axis, dtype=float)
    second = np.asarray(frame.v_axis, dtype=float)
    return (
        (float(first[0]), float(second[0]), float(origin[0])),
        (float(first[1]), float(second[1]), float(origin[1])),
        (float(first[2]), float(second[2]), float(origin[2])),
        (0.0, 0.0, 1.0),
    )


@dataclass(frozen=True, slots=True)
class DandelinDiagramPoint2D:
    """One stable semantic point with both plane and world coordinates."""

    point_id: str
    source_ref: str
    point: PlanarPoint3D

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_id", _identity(self.point_id, "point_id"))
        object.__setattr__(
            self,
            "source_ref",
            _identity(self.source_ref, "source_ref"),
        )
        if not isinstance(self.point, PlanarPoint3D):
            raise TypeError("point must be a PlanarPoint3D")

    @property
    def frame(self) -> PlanarFrame3D:
        return self.point.frame

    @property
    def coordinates(self) -> tuple[float, float]:
        return self.point.coordinates

    @property
    def world_point(self) -> tuple[float, float, float]:
        return self.point.world_point

    def to_dict(self) -> dict[str, object]:
        return {
            "pointId": self.point_id,
            "sourceRef": self.source_ref,
            "frameId": self.frame.frame_id,
            "coordinates": list(self.coordinates),
            "worldPoint": list(self.world_point),
        }


@dataclass(frozen=True, slots=True)
class DandelinDiagramLine2D:
    """One infinite line carried by a diagram frame."""

    line_id: str
    source_ref: str
    point: PlanarPoint3D
    direction_coordinates: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "line_id", _identity(self.line_id, "line_id"))
        object.__setattr__(
            self,
            "source_ref",
            _identity(self.source_ref, "source_ref"),
        )
        if not isinstance(self.point, PlanarPoint3D):
            raise TypeError("line point must be a PlanarPoint3D")
        object.__setattr__(
            self,
            "direction_coordinates",
            _canonical_direction2(self.direction_coordinates, "line direction"),
        )

    @property
    def frame(self) -> PlanarFrame3D:
        return self.point.frame

    @property
    def point_coordinates(self) -> tuple[float, float]:
        return self.point.coordinates

    @property
    def world_point(self) -> tuple[float, float, float]:
        return self.point.world_point

    @property
    def world_direction(self) -> tuple[float, float, float]:
        direction = (
            self.direction_coordinates[0] * np.asarray(self.frame.u_axis, dtype=float)
            + self.direction_coordinates[1]
            * np.asarray(self.frame.v_axis, dtype=float)
        )
        return _point3(direction, "line world direction")

    def to_dict(self) -> dict[str, object]:
        return {
            "lineId": self.line_id,
            "sourceRef": self.source_ref,
            "frameId": self.frame.frame_id,
            "pointCoordinates": list(self.point_coordinates),
            "directionCoordinates": list(self.direction_coordinates),
            "worldPoint": list(self.world_point),
            "worldDirection": list(self.world_direction),
        }


@dataclass(frozen=True, slots=True)
class DandelinDiagramSegment2D:
    """One finite line segment carried by a diagram frame."""

    segment_id: str
    source_ref: str
    start: PlanarPoint3D
    end: PlanarPoint3D

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "segment_id",
            _identity(self.segment_id, "segment_id"),
        )
        object.__setattr__(
            self,
            "source_ref",
            _identity(self.source_ref, "source_ref"),
        )
        if not isinstance(self.start, PlanarPoint3D) or not isinstance(
            self.end, PlanarPoint3D
        ):
            raise TypeError("segment endpoints must be PlanarPoint3D objects")
        if self.start.frame != self.end.frame:
            raise DandelinView2DError(
                "segment endpoints must use the same diagram frame"
            )
        displacement = np.asarray(self.end.coordinates) - np.asarray(
            self.start.coordinates
        )
        length = float(np.linalg.norm(displacement))
        if not isfinite(length) or length <= 0.0:
            raise DandelinView2DError("diagram segment must have positive length")

    @property
    def frame(self) -> PlanarFrame3D:
        return self.start.frame

    @property
    def direction_coordinates(self) -> tuple[float, float]:
        return _unit2(
            np.asarray(self.end.coordinates) - np.asarray(self.start.coordinates),
            "segment direction",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "segmentId": self.segment_id,
            "sourceRef": self.source_ref,
            "frameId": self.frame.frame_id,
            "startCoordinates": list(self.start.coordinates),
            "endCoordinates": list(self.end.coordinates),
            "startWorldPoint": list(self.start.world_point),
            "endWorldPoint": list(self.end.world_point),
        }


@dataclass(frozen=True, slots=True)
class DandelinSphereCircleSection2D:
    """The genuine great-circle section of one sphere by a meridian plane."""

    circle_id: str
    source_ref: str
    sphere_id: str
    center: PlanarPoint3D
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "circle_id", _identity(self.circle_id, "circle_id"))
        object.__setattr__(
            self,
            "source_ref",
            _identity(self.source_ref, "source_ref"),
        )
        object.__setattr__(self, "sphere_id", _identity(self.sphere_id, "sphere_id"))
        if not isinstance(self.center, PlanarPoint3D):
            raise TypeError("circle center must be a PlanarPoint3D")
        object.__setattr__(self, "radius", _positive(self.radius, "circle radius"))

    @property
    def frame(self) -> PlanarFrame3D:
        return self.center.frame

    @property
    def center_coordinates(self) -> tuple[float, float]:
        return self.center.coordinates

    @property
    def world_center(self) -> tuple[float, float, float]:
        return self.center.world_point

    def to_dict(self) -> dict[str, object]:
        return {
            "circleId": self.circle_id,
            "sourceRef": self.source_ref,
            "sphereId": self.sphere_id,
            "frameId": self.frame.frame_id,
            "centerCoordinates": list(self.center_coordinates),
            "worldCenter": list(self.world_center),
            "radius": self.radius,
        }


@dataclass(frozen=True, slots=True)
class DandelinCircleLineTangency2D:
    """Numerical certificate for one circle/line tangency in the meridian."""

    tangency_id: str
    source_ref: str
    sphere_id: str
    circle_id: str
    carrier_id: str
    contact: DandelinDiagramPoint2D
    circle_residual: float
    carrier_residual: float
    orthogonality_residual: float

    def __post_init__(self) -> None:
        for name in (
            "tangency_id",
            "source_ref",
            "sphere_id",
            "circle_id",
            "carrier_id",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        if not isinstance(self.contact, DandelinDiagramPoint2D):
            raise TypeError("contact must be a DandelinDiagramPoint2D")
        for name in (
            "circle_residual",
            "carrier_residual",
            "orthogonality_residual",
        ):
            object.__setattr__(
                self,
                name,
                _finite_residual(getattr(self, name), name),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "tangencyId": self.tangency_id,
            "sourceRef": self.source_ref,
            "sphereId": self.sphere_id,
            "circleId": self.circle_id,
            "carrierId": self.carrier_id,
            "contact": self.contact.to_dict(),
            "circleResidual": self.circle_residual,
            "carrierResidual": self.carrier_residual,
            "orthogonalityResidual": self.orthogonality_residual,
        }


@dataclass(frozen=True, slots=True)
class DandelinSpherePlaneTangency2D:
    """Sphere-plane contact evidence presented in cutting-plane coordinates.

    The sphere centre remains a world-space point; it is intentionally not
    projected into the section plane and must not be interpreted as the centre
    of a circle in this diagram.
    """

    tangency_id: str
    source_ref: str
    sphere_id: str
    focus: DandelinDiagramPoint2D
    sphere_center_world: tuple[float, float, float]
    sphere_radius: float
    sphere_residual: float
    plane_residual: float
    normal_alignment_residual: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tangency_id",
            _identity(self.tangency_id, "tangency_id"),
        )
        object.__setattr__(
            self,
            "source_ref",
            _identity(self.source_ref, "source_ref"),
        )
        object.__setattr__(self, "sphere_id", _identity(self.sphere_id, "sphere_id"))
        if not isinstance(self.focus, DandelinDiagramPoint2D):
            raise TypeError("focus must be a DandelinDiagramPoint2D")
        object.__setattr__(
            self,
            "sphere_center_world",
            _point3(self.sphere_center_world, "sphere_center_world"),
        )
        object.__setattr__(
            self,
            "sphere_radius",
            _positive(self.sphere_radius, "sphere_radius"),
        )
        for name in (
            "sphere_residual",
            "plane_residual",
            "normal_alignment_residual",
        ):
            object.__setattr__(
                self,
                name,
                _finite_residual(getattr(self, name), name),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "tangencyId": self.tangency_id,
            "sourceRef": self.source_ref,
            "sphereId": self.sphere_id,
            "focus": self.focus.to_dict(),
            "sphereCenterWorld": list(self.sphere_center_world),
            "sphereRadius": self.sphere_radius,
            "sphereResidual": self.sphere_residual,
            "planeResidual": self.plane_residual,
            "normalAlignmentResidual": self.normal_alignment_residual,
        }


@dataclass(frozen=True, slots=True)
class DandelinMeridianDiagram2D:
    """Exact cone-axis meridian view with genuine sphere circle sections."""

    diagram_id: str
    construction: DandelinConstruction3D
    frame: PlanarFrame3D
    radial_source: str
    section_line: DandelinDiagramLine2D
    generators: tuple[DandelinDiagramSegment2D, ...]
    sphere_circles: tuple[DandelinSphereCircleSection2D, ...]
    focus_points: tuple[DandelinDiagramPoint2D, ...]
    tangencies: tuple[DandelinCircleLineTangency2D, ...]
    certification_tolerance: float
    angular_tolerance: float
    schema: str = DANDELIN_MERIDIAN_DIAGRAM_2D_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DANDELIN_MERIDIAN_DIAGRAM_2D_SCHEMA:
            raise DandelinView2DError("invalid Dandelin-meridian diagram schema")
        object.__setattr__(self, "diagram_id", _identity(self.diagram_id, "diagram_id"))
        if not isinstance(self.construction, DandelinConstruction3D):
            raise TypeError("construction must be a DandelinConstruction3D")
        if not isinstance(self.frame, PlanarFrame3D):
            raise TypeError("frame must be a PlanarFrame3D")
        if self.radial_source not in {"projected_plane_normal", "cone_radial_axis"}:
            raise DandelinView2DError(
                "radial_source must be projected_plane_normal or cone_radial_axis"
            )
        if not isinstance(self.section_line, DandelinDiagramLine2D):
            raise TypeError("section_line must be a DandelinDiagramLine2D")
        generators = tuple(self.generators)
        circles = tuple(self.sphere_circles)
        foci = tuple(self.focus_points)
        tangencies = tuple(self.tangencies)
        if not generators or not all(
            isinstance(item, DandelinDiagramSegment2D) for item in generators
        ):
            raise DandelinView2DError("a meridian diagram requires cone generators")
        if not circles or not all(
            isinstance(item, DandelinSphereCircleSection2D) for item in circles
        ):
            raise DandelinView2DError(
                "a meridian diagram requires genuine sphere circle sections"
            )
        if not all(isinstance(item, DandelinDiagramPoint2D) for item in foci):
            raise TypeError("focus_points must contain DandelinDiagramPoint2D values")
        if not all(
            isinstance(item, DandelinCircleLineTangency2D) for item in tangencies
        ):
            raise TypeError(
                "tangencies must contain DandelinCircleLineTangency2D values"
            )
        for item in (self.section_line, *generators, *circles, *foci):
            if item.frame != self.frame:
                raise DandelinView2DError(
                    "all meridian geometry must use the meridian diagram frame"
                )
        _sorted_unique_identities(
            tuple(item.segment_id for item in generators),
            "meridian generators",
        )
        _sorted_unique_identities(
            tuple(item.circle_id for item in circles),
            "meridian sphere circles",
        )
        _sorted_unique_identities(
            tuple(item.point_id for item in foci),
            "meridian focus points",
        )
        _sorted_unique_identities(
            tuple(item.tangency_id for item in tangencies),
            "meridian tangencies",
        )
        length_tolerance = _positive(
            self.certification_tolerance,
            "certification_tolerance",
        )
        angular_tolerance = _positive(self.angular_tolerance, "angular_tolerance")
        circle_ids = {item.circle_id for item in circles}
        sphere_ids = {item.sphere_id for item in circles}
        if len(sphere_ids) != len(circles):
            raise DandelinView2DError(
                "each meridian sphere must own exactly one circle section"
            )
        circle_map = {item.circle_id: item for item in circles}
        generator_map = {item.segment_id: item for item in generators}
        focus_map = {item.point_id: item for item in foci}
        carrier_ids = {
            self.section_line.line_id,
            *(item.segment_id for item in generators),
        }
        if {item.sphere_id for item in tangencies} != sphere_ids:
            raise DandelinView2DError(
                "meridian tangencies do not cover every sphere circle"
            )
        section_contact_ids = {
            item.contact.point_id
            for item in tangencies
            if item.carrier_id == self.section_line.line_id
        }
        if section_contact_ids != set(focus_map):
            raise DandelinView2DError(
                "meridian focus points must be exactly the section-line contacts"
            )
        if any(item.circle_id not in circle_ids for item in tangencies):
            raise DandelinView2DError(
                "a meridian tangency references an unknown circle"
            )
        if any(item.carrier_id not in carrier_ids for item in tangencies):
            raise DandelinView2DError(
                "a meridian tangency references an unknown line or generator"
            )
        for evidence in tangencies:
            if evidence.contact.frame != self.frame:
                raise DandelinView2DError(
                    "every meridian contact point must use the diagram frame"
                )
            circle = circle_map[evidence.circle_id]
            if circle.sphere_id != evidence.sphere_id:
                raise DandelinView2DError(
                    "meridian tangency sphere_id does not own its referenced circle"
                )
            if evidence.carrier_id == self.section_line.line_id:
                carrier_anchor = self.section_line.point_coordinates
                carrier_direction = self.section_line.direction_coordinates
                focus = focus_map.get(evidence.contact.point_id)
                if focus is None or focus != evidence.contact:
                    raise DandelinView2DError(
                        "section-line tangency contact must be the sphere focus"
                    )
            else:
                generator = generator_map[evidence.carrier_id]
                carrier_anchor = generator.start.coordinates
                carrier_direction = generator.direction_coordinates
                contact_coordinates = np.asarray(
                    evidence.contact.coordinates,
                    dtype=float,
                )
                start_coordinates = np.asarray(generator.start.coordinates, dtype=float)
                end_coordinates = np.asarray(generator.end.coordinates, dtype=float)
                segment_vector = end_coordinates - start_coordinates
                segment_length = float(np.linalg.norm(segment_vector))
                along = float(
                    np.dot(
                        contact_coordinates - start_coordinates,
                        segment_vector / segment_length,
                    )
                )
                if along < -length_tolerance or along > segment_length + length_tolerance:
                    raise DandelinView2DError(
                        "generator tangency contact lies outside its finite segment"
                    )
            recomputed = _circle_line_evidence(
                evidence.tangency_id,
                evidence.source_ref,
                evidence.sphere_id,
                circle,
                evidence.carrier_id,
                carrier_anchor,
                carrier_direction,
                evidence.contact,
            )
            if not all(
                _same_residual(actual, recorded)
                for actual, recorded in (
                    (recomputed.circle_residual, evidence.circle_residual),
                    (recomputed.carrier_residual, evidence.carrier_residual),
                    (
                        recomputed.orthogonality_residual,
                        evidence.orthogonality_residual,
                    ),
                )
            ):
                raise DandelinView2DError(
                    f"tangency {evidence.tangency_id!r} residual evidence is stale or forged"
                )
        for sphere_id in sphere_ids:
            records = tuple(item for item in tangencies if item.sphere_id == sphere_id)
            if len(records) != 3:
                raise DandelinView2DError(
                    "each meridian sphere requires one section-line and two "
                    "generator tangencies"
                )
            if sum(item.carrier_id == self.section_line.line_id for item in records) != 1:
                raise DandelinView2DError(
                    "each meridian sphere requires exactly one section-line tangency"
                )
            generator_carriers = {
                item.carrier_id
                for item in records
                if item.carrier_id != self.section_line.line_id
            }
            if len(generator_carriers) != 2:
                raise DandelinView2DError(
                    "each meridian sphere requires two distinct generator tangencies"
                )
        for evidence in tangencies:
            if (
                evidence.circle_residual > length_tolerance
                or evidence.carrier_residual > length_tolerance
                or evidence.orthogonality_residual > angular_tolerance
            ):
                raise DandelinView2DError(
                    f"tangency {evidence.tangency_id!r} exceeds its certification tolerance"
                )
        expected = _derive_dandelin_meridian_parts(self.construction)
        if (
            self.diagram_id,
            self.frame,
            self.radial_source,
            self.section_line,
            generators,
            circles,
            foci,
            tangencies,
            length_tolerance,
            angular_tolerance,
        ) != (
            expected.diagram_id,
            expected.frame,
            expected.radial_source,
            expected.section_line,
            expected.generators,
            expected.sphere_circles,
            expected.focus_points,
            expected.tangencies,
            expected.certification_tolerance,
            expected.angular_tolerance,
        ):
            raise DandelinView2DError(
                "meridian diagram does not match its authoritative construction"
            )
        object.__setattr__(self, "generators", generators)
        object.__setattr__(self, "sphere_circles", circles)
        object.__setattr__(self, "focus_points", foci)
        object.__setattr__(self, "tangencies", tangencies)
        object.__setattr__(self, "certification_tolerance", length_tolerance)
        object.__setattr__(self, "angular_tolerance", angular_tolerance)

    @property
    def construction_id(self) -> str:
        return self.construction.construction_id

    @property
    def world_embedding(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """Affine map from homogeneous ``(u, v, 1)`` to world coordinates."""

        return _world_embedding(self.frame)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "diagramId": self.diagram_id,
            "constructionId": self.construction_id,
            "frame": self.frame.to_dict(),
            "worldEmbedding": [list(row) for row in self.world_embedding],
            "radialSource": self.radial_source,
            "sectionLine": self.section_line.to_dict(),
            "generators": [item.to_dict() for item in self.generators],
            "sphereCircles": [item.to_dict() for item in self.sphere_circles],
            "focusPoints": [item.to_dict() for item in self.focus_points],
            "tangencies": [item.to_dict() for item in self.tangencies],
            "certificationTolerance": self.certification_tolerance,
            "angularTolerance": self.angular_tolerance,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            _normalize_canonical_numbers(self.to_dict()),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class DandelinSectionPlaneDiagram2D:
    """The exact conic view in the authored cutting plane.

    This type intentionally has no ``sphere_circles`` or ``circles`` field.
    Only sphere-plane contact evidence crosses into the cutting-plane view.
    """

    diagram_id: str
    construction: DandelinConstruction3D
    frame: PlanarFrame3D
    supporting_kind: ConicKind
    conic_trace: QuadricSectionTrace
    focus_points: tuple[DandelinDiagramPoint2D, ...]
    directrices: tuple[DandelinDiagramLine2D, ...]
    sphere_plane_tangencies: tuple[DandelinSpherePlaneTangency2D, ...]
    certification_tolerance: float
    angular_tolerance: float
    schema: str = DANDELIN_SECTION_PLANE_DIAGRAM_2D_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DANDELIN_SECTION_PLANE_DIAGRAM_2D_SCHEMA:
            raise DandelinView2DError(
                "invalid Dandelin section-plane diagram schema"
            )
        object.__setattr__(self, "diagram_id", _identity(self.diagram_id, "diagram_id"))
        if not isinstance(self.construction, DandelinConstruction3D):
            raise TypeError("construction must be a DandelinConstruction3D")
        if not isinstance(self.frame, PlanarFrame3D):
            raise TypeError("frame must be a PlanarFrame3D")
        if not isinstance(self.supporting_kind, ConicKind):
            raise TypeError("supporting_kind must be a ConicKind")
        if not isinstance(self.conic_trace, QuadricSectionTrace):
            raise TypeError("conic_trace must be a QuadricSectionTrace")
        if self.conic_trace.supporting_kind is not self.supporting_kind:
            raise DandelinView2DError(
                "conic_trace kind does not match the section-plane diagram"
            )
        if self.conic_trace.section_id != f"{self.diagram_id}:conic":
            raise DandelinView2DError(
                "conic_trace identity does not match the section-plane diagram"
            )
        foci = tuple(self.focus_points)
        directrices = tuple(self.directrices)
        tangencies = tuple(self.sphere_plane_tangencies)
        if not foci or not all(
            isinstance(item, DandelinDiagramPoint2D) for item in foci
        ):
            raise DandelinView2DError(
                "a section-plane diagram requires certified focus points"
            )
        if not all(isinstance(item, DandelinDiagramLine2D) for item in directrices):
            raise TypeError("directrices must contain DandelinDiagramLine2D values")
        if not all(
            isinstance(item, DandelinSpherePlaneTangency2D) for item in tangencies
        ):
            raise TypeError(
                "sphere_plane_tangencies must contain DandelinSpherePlaneTangency2D values"
            )
        for item in (*foci, *directrices):
            if item.frame != self.frame:
                raise DandelinView2DError(
                    "all section-plane geometry must use the authored section frame"
                )
        _sorted_unique_identities(
            tuple(item.point_id for item in foci),
            "section-plane focus points",
        )
        _sorted_unique_identities(
            tuple(item.line_id for item in directrices),
            "section-plane directrices",
        )
        _sorted_unique_identities(
            tuple(item.tangency_id for item in tangencies),
            "section-plane tangencies",
        )
        if {item.focus.point_id for item in tangencies} != {
            item.point_id for item in foci
        }:
            raise DandelinView2DError(
                "sphere-plane tangencies do not match the diagram focus points"
            )
        if len(tangencies) != len(foci):
            raise DandelinView2DError(
                "each section-plane focus requires exactly one sphere tangency"
            )
        expected_directrix_count = {
            ConicKind.CIRCLE: 0,
            ConicKind.ELLIPSE: 2,
            ConicKind.PARABOLA: 1,
            ConicKind.HYPERBOLA: 2,
        }.get(self.supporting_kind)
        if expected_directrix_count is None or len(directrices) != expected_directrix_count:
            raise DandelinView2DError(
                "section-plane directrix count does not match the conic kind"
            )
        focus_map = {item.point_id: item for item in foci}
        if len({item.sphere_id for item in tangencies}) != len(tangencies):
            raise DandelinView2DError(
                "each section-plane tangency must reference a distinct sphere"
            )
        length_tolerance = _positive(
            self.certification_tolerance,
            "certification_tolerance",
        )
        angular_tolerance = _positive(self.angular_tolerance, "angular_tolerance")
        embedding_error = float(
            np.max(
                np.abs(
                    np.asarray(self.conic_trace.plane_embedding, dtype=float)
                    - np.asarray(_world_embedding(self.frame), dtype=float)
                )
            )
        )
        if not isfinite(embedding_error) or embedding_error > length_tolerance:
            raise DandelinView2DError(
                "conic_trace plane embedding does not match the authored section frame"
            )
        for evidence in tangencies:
            if (
                evidence.focus.frame != self.frame
                or focus_map[evidence.focus.point_id] != evidence.focus
            ):
                raise DandelinView2DError(
                    "sphere-plane tangency focus does not match the diagram focus"
                )
            center = np.asarray(evidence.sphere_center_world, dtype=float)
            focus_world = np.asarray(evidence.focus.world_point, dtype=float)
            radius_vector = focus_world - center
            radius_length = float(np.linalg.norm(radius_vector))
            if not isfinite(radius_length) or radius_length <= 0.0:
                raise DandelinView2DError(
                    "sphere-plane tangency radius vector must be non-zero"
                )
            frame_normal = np.asarray(self.construction.plane.normal, dtype=float)
            recomputed = (
                abs(radius_length - evidence.sphere_radius),
                abs(self.construction.plane.signed_distance(focus_world)),
                float(
                    np.linalg.norm(
                        np.cross(radius_vector / radius_length, frame_normal)
                    )
                ),
            )
            recorded = (
                evidence.sphere_residual,
                evidence.plane_residual,
                evidence.normal_alignment_residual,
            )
            if not all(
                _same_residual(actual, claimed)
                for actual, claimed in zip(recomputed, recorded)
            ):
                raise DandelinView2DError(
                    f"tangency {evidence.tangency_id!r} residual evidence is stale or forged"
                )
            if (
                recomputed[0] > length_tolerance
                or recomputed[1] > length_tolerance
                or recomputed[2] > angular_tolerance
            ):
                raise DandelinView2DError(
                    f"tangency {evidence.tangency_id!r} exceeds its certification tolerance"
                )
        expected = _derive_dandelin_section_plane_parts(self.construction)
        if (
            self.diagram_id,
            self.frame,
            self.supporting_kind,
            self.conic_trace,
            foci,
            directrices,
            tangencies,
            length_tolerance,
            angular_tolerance,
        ) != (
            expected.diagram_id,
            expected.frame,
            expected.supporting_kind,
            expected.conic_trace,
            expected.focus_points,
            expected.directrices,
            expected.sphere_plane_tangencies,
            expected.certification_tolerance,
            expected.angular_tolerance,
        ):
            raise DandelinView2DError(
                "section-plane diagram does not match its authoritative construction"
            )
        object.__setattr__(self, "focus_points", foci)
        object.__setattr__(self, "directrices", directrices)
        object.__setattr__(self, "sphere_plane_tangencies", tangencies)
        object.__setattr__(self, "certification_tolerance", length_tolerance)
        object.__setattr__(self, "angular_tolerance", angular_tolerance)

    @property
    def construction_id(self) -> str:
        return self.construction.construction_id

    @property
    def world_embedding(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """Affine map from homogeneous ``(u, v, 1)`` to world coordinates."""

        return _world_embedding(self.frame)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "diagramId": self.diagram_id,
            "constructionId": self.construction_id,
            "frame": self.frame.to_dict(),
            "worldEmbedding": [list(row) for row in self.world_embedding],
            "supportingKind": self.supporting_kind.value,
            "conicTrace": self.conic_trace.to_dict(),
            "focusPoints": [item.to_dict() for item in self.focus_points],
            "directrices": [item.to_dict() for item in self.directrices],
            "spherePlaneTangencies": [
                item.to_dict() for item in self.sphere_plane_tangencies
            ],
            "certificationTolerance": self.certification_tolerance,
            "angularTolerance": self.angular_tolerance,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            _normalize_canonical_numbers(self.to_dict()),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _view_tolerances(construction: DandelinConstruction3D) -> tuple[float, float]:
    context = construction.certification_context
    length = max(
        64.0 * context.epsilon(GeometryQuantity.BOUNDARY),
        512.0 * np.finfo(float).eps * context.resolved.scale,
    )
    angular = max(
        64.0 * context.epsilon(GeometryQuantity.ANGULAR),
        512.0 * np.finfo(float).eps,
    )
    return _positive(length, "view length tolerance"), _positive(
        angular,
        "view angular tolerance",
    )


def _reembed_world_point(
    point_id: str,
    source_ref: str,
    frame: PlanarFrame3D,
    world_point: Sequence[float],
    *,
    tolerance: float,
) -> DandelinDiagramPoint2D:
    world = np.asarray(_point3(world_point, f"{point_id} world point"), dtype=float)
    origin = np.asarray(frame.point, dtype=float)
    delta = world - origin
    coordinates = (
        float(np.dot(delta, np.asarray(frame.u_axis, dtype=float))),
        float(np.dot(delta, np.asarray(frame.v_axis, dtype=float))),
    )
    point = frame.certified_point(coordinates)
    error = float(
        np.linalg.norm(np.asarray(point.world_point, dtype=float) - world)
    )
    if not isfinite(error) or error > tolerance:
        raise DandelinView2DError(
            f"point {point_id!r} does not lie in the requested two-dimensional view"
        )
    return DandelinDiagramPoint2D(point_id, source_ref, point)


def _line_residual(
    point: Sequence[float],
    anchor: Sequence[float],
    direction: Sequence[float],
) -> float:
    value = np.asarray(_point2(point, "line query point"), dtype=float)
    origin = np.asarray(_point2(anchor, "line anchor"), dtype=float)
    unit = np.asarray(_unit2(direction, "line direction"), dtype=float)
    delta = value - origin
    return abs(float(delta[0] * unit[1] - delta[1] * unit[0]))


def _circle_line_evidence(
    tangency_id: str,
    source_ref: str,
    sphere_id: str,
    circle: DandelinSphereCircleSection2D,
    carrier_id: str,
    carrier_anchor: Sequence[float],
    carrier_direction: Sequence[float],
    contact: DandelinDiagramPoint2D,
) -> DandelinCircleLineTangency2D:
    center = np.asarray(circle.center_coordinates, dtype=float)
    point = np.asarray(contact.coordinates, dtype=float)
    radius_vector = point - center
    radius_length = float(np.linalg.norm(radius_vector))
    if not isfinite(radius_length) or radius_length <= 0.0:
        raise DandelinView2DError("circle tangency radius vector must be non-zero")
    tangent = np.asarray(_unit2(carrier_direction, "carrier direction"), dtype=float)
    return DandelinCircleLineTangency2D(
        tangency_id=tangency_id,
        source_ref=source_ref,
        sphere_id=sphere_id,
        circle_id=circle.circle_id,
        carrier_id=carrier_id,
        contact=contact,
        circle_residual=abs(radius_length - circle.radius),
        carrier_residual=_line_residual(
            contact.coordinates,
            carrier_anchor,
            tangent,
        ),
        orthogonality_residual=abs(
            float(np.dot(radius_vector / radius_length, tangent))
        ),
    )


def _sphere_plane_evidence(
    construction: DandelinConstruction3D,
    record: DandelinSphere3D,
    focus: DandelinDiagramPoint2D,
) -> DandelinSpherePlaneTangency2D:
    sphere = record.sphere
    center = np.asarray(sphere.center, dtype=float)
    point = np.asarray(focus.world_point, dtype=float)
    radius_vector = point - center
    radius_length = float(np.linalg.norm(radius_vector))
    if not isfinite(radius_length) or radius_length <= 0.0:
        raise DandelinView2DError(
            "sphere-plane tangency radius vector must be non-zero"
        )
    normal = np.asarray(construction.plane.normal, dtype=float)
    return DandelinSpherePlaneTangency2D(
        tangency_id=f"{focus.point_id}:sphere-plane-tangency",
        source_ref=record.focus_id,
        sphere_id=record.sphere_id,
        focus=focus,
        sphere_center_world=_point3(center, "sphere center"),
        sphere_radius=sphere.radius,
        sphere_residual=abs(radius_length - sphere.radius),
        plane_residual=abs(construction.plane.signed_distance(point)),
        normal_alignment_residual=float(
            np.linalg.norm(np.cross(radius_vector / radius_length, normal))
        ),
    )


def _canonical_meridian_radial(
    construction: DandelinConstruction3D,
    *,
    angular_tolerance: float,
) -> tuple[np.ndarray, str]:
    axis = np.asarray(construction.cone.axis, dtype=float)
    normal = np.asarray(construction.plane.normal, dtype=float)
    projected = normal - float(np.dot(normal, axis)) * axis
    length = float(np.linalg.norm(projected))
    if not isfinite(length):
        raise DandelinView2DError(
            "section-plane normal cannot define a finite meridian direction"
        )
    if length <= angular_tolerance:
        if construction.supporting_kind is not ConicKind.CIRCLE:
            raise DandelinView2DError(
                "a non-circular section has no certifiable projected-normal "
                "meridian direction"
            )
        radial = np.asarray(construction.cone.radial_axis, dtype=float)
        source = "cone_radial_axis"
    else:
        radial = projected / length
        source = "projected_plane_normal"
        cone_frame = construction.cone.frame
        for reference in (cone_frame.x_axis, cone_frame.y_axis):
            orientation = float(np.dot(radial, np.asarray(reference, dtype=float)))
            if abs(orientation) > angular_tolerance:
                if orientation < 0.0:
                    radial = -radial
                break
        else:  # pragma: no cover - an orthonormal radial basis makes this impossible
            raise DandelinView2DError(
                "projected-normal meridian direction has no canonical orientation"
            )
    radial -= float(np.dot(radial, axis)) * axis
    radial_length = float(np.linalg.norm(radial))
    if not isfinite(radial_length) or radial_length <= angular_tolerance:
        raise DandelinView2DError(
            "meridian radial direction is parallel to the cone axis"
        )
    return radial / radial_length, source


@dataclass(frozen=True, slots=True)
class _DandelinMeridianParts:
    diagram_id: str
    frame: PlanarFrame3D
    radial_source: str
    section_line: DandelinDiagramLine2D
    generators: tuple[DandelinDiagramSegment2D, ...]
    sphere_circles: tuple[DandelinSphereCircleSection2D, ...]
    focus_points: tuple[DandelinDiagramPoint2D, ...]
    tangencies: tuple[DandelinCircleLineTangency2D, ...]
    certification_tolerance: float
    angular_tolerance: float


@dataclass(frozen=True, slots=True)
class _DandelinSectionPlaneParts:
    diagram_id: str
    frame: PlanarFrame3D
    supporting_kind: ConicKind
    conic_trace: QuadricSectionTrace
    focus_points: tuple[DandelinDiagramPoint2D, ...]
    directrices: tuple[DandelinDiagramLine2D, ...]
    sphere_plane_tangencies: tuple[DandelinSpherePlaneTangency2D, ...]
    certification_tolerance: float
    angular_tolerance: float


def _derive_dandelin_meridian_parts(
    construction: DandelinConstruction3D,
) -> _DandelinMeridianParts:
    """Derive the true cone-axis meridian diagram from one construction."""

    if not isinstance(construction, DandelinConstruction3D):
        raise TypeError("construction must be a DandelinConstruction3D")
    length_tolerance, angular_tolerance = _view_tolerances(construction)
    diagram_id = f"{construction.construction_id}:view:meridian"
    axis = np.asarray(construction.cone.axis, dtype=float)
    radial, radial_source = _canonical_meridian_radial(
        construction,
        angular_tolerance=angular_tolerance,
    )
    meridian_normal = np.cross(radial, axis)
    try:
        frame = PlanarFrame3D(
            f"{diagram_id}:frame",
            construction.cone.apex,
            meridian_normal,
            radial,
        )
    except (TypeError, ValueError) as exc:
        raise DandelinView2DError(
            f"meridian frame cannot be certified: {exc}"
        ) from exc
    if (
        float(np.linalg.norm(np.asarray(frame.u_axis) - radial))
        > angular_tolerance
        or float(np.linalg.norm(np.asarray(frame.v_axis) - axis))
        > angular_tolerance
    ):
        raise DandelinView2DError(
            "meridian frame does not preserve radial/axis coordinates"
        )

    plane_normal = np.asarray(construction.plane.normal, dtype=float)
    coefficients = np.asarray(
        (
            float(np.dot(plane_normal, radial)),
            float(np.dot(plane_normal, axis)),
        ),
        dtype=float,
    )
    coefficient_length = float(np.linalg.norm(coefficients))
    if not isfinite(coefficient_length) or coefficient_length <= angular_tolerance:
        raise DandelinView2DError(
            "section plane has no certifiable line in the meridian plane"
        )
    constant = float(
        np.dot(
            plane_normal,
            np.asarray(construction.cone.apex, dtype=float)
            - np.asarray(construction.plane.point, dtype=float),
        )
    )
    section_anchor_coordinates = -constant * coefficients / (
        coefficient_length * coefficient_length
    )
    section_direction = _canonical_direction2(
        (-coefficients[1], coefficients[0]),
        "meridian section-line direction",
    )
    section_line = DandelinDiagramLine2D(
        f"{diagram_id}:section-line",
        construction.plane.plane_id,
        frame.certified_point(section_anchor_coordinates),
        section_direction,
    )

    generator_records: list[DandelinDiagramSegment2D] = []
    lower, upper = construction.cone.axial_range
    nappe_intervals = []
    if lower < 0.0:
        nappe_intervals.append(("negative", lower, min(upper, 0.0)))
    if upper > 0.0:
        nappe_intervals.append(("positive", max(lower, 0.0), upper))
    for nappe_label, start_axial, end_axial in nappe_intervals:
        if end_axial - start_axial <= length_tolerance:
            raise DandelinView2DError(
                f"cone nappe {nappe_label!r} has no finite meridian extent"
            )
        for side_label, side_sign in (("negative", -1.0), ("positive", 1.0)):
            start_coordinates = (
                side_sign * abs(start_axial) * construction.cone.slope,
                start_axial,
            )
            end_coordinates = (
                side_sign * abs(end_axial) * construction.cone.slope,
                end_axial,
            )
            generator_records.append(
                DandelinDiagramSegment2D(
                    f"{diagram_id}:generator:nappe:{nappe_label}:side:{side_label}",
                    construction.cone.surface_id,
                    frame.certified_point(start_coordinates),
                    frame.certified_point(end_coordinates),
                )
            )
    generators = tuple(sorted(generator_records, key=lambda item: item.segment_id))
    generator_map = {item.segment_id: item for item in generators}

    circles: list[DandelinSphereCircleSection2D] = []
    foci: list[DandelinDiagramPoint2D] = []
    tangencies: list[DandelinCircleLineTangency2D] = []
    for record in construction.spheres:
        center = _reembed_world_point(
            f"{record.sphere_id}:meridian-center",
            record.sphere_id,
            frame,
            record.sphere.center,
            tolerance=length_tolerance,
        ).point
        circle = DandelinSphereCircleSection2D(
            f"{record.sphere_id}:meridian-circle",
            record.sphere_id,
            record.sphere_id,
            center,
            record.sphere.radius,
        )
        circles.append(circle)

        focus = _reembed_world_point(
            f"{diagram_id}:point:{record.focus_id}",
            record.focus_id,
            frame,
            record.focus.world_point,
            tolerance=length_tolerance,
        )
        foci.append(focus)
        tangencies.append(
            _circle_line_evidence(
                f"{focus.point_id}:section-tangency",
                record.focus_id,
                record.sphere_id,
                circle,
                section_line.line_id,
                section_line.point_coordinates,
                section_line.direction_coordinates,
                focus,
            )
        )

        contact_center = _reembed_world_point(
            f"{record.cone_contact_circle.curve_id}:meridian-center",
            record.cone_contact_circle.curve_id,
            frame,
            record.cone_contact_circle.center,
            tolerance=length_tolerance,
        )
        nappe_label = "positive" if record.nappe_sign > 0 else "negative"
        for side_label, side_sign in (("negative", -1.0), ("positive", 1.0)):
            contact_coordinates = np.asarray(contact_center.coordinates, dtype=float)
            contact_coordinates[0] += side_sign * record.cone_contact_circle.radius
            point_id = (
                f"{diagram_id}:point:{record.cone_contact_circle.curve_id}:"
                f"side:{side_label}"
            )
            contact = DandelinDiagramPoint2D(
                point_id,
                record.cone_contact_circle.curve_id,
                frame.certified_point(contact_coordinates),
            )
            generator_id = (
                f"{diagram_id}:generator:nappe:{nappe_label}:side:{side_label}"
            )
            generator = generator_map.get(generator_id)
            if generator is None:
                raise DandelinView2DError(
                    f"sphere {record.sphere_id!r} references a missing cone generator"
                )
            tangencies.append(
                _circle_line_evidence(
                    f"{point_id}:tangency",
                    record.cone_contact_circle.curve_id,
                    record.sphere_id,
                    circle,
                    generator.segment_id,
                    generator.start.coordinates,
                    generator.direction_coordinates,
                    contact,
                )
            )

    return _DandelinMeridianParts(
        diagram_id=diagram_id,
        frame=frame,
        radial_source=radial_source,
        section_line=section_line,
        generators=generators,
        sphere_circles=tuple(sorted(circles, key=lambda item: item.circle_id)),
        focus_points=tuple(sorted(foci, key=lambda item: item.point_id)),
        tangencies=tuple(sorted(tangencies, key=lambda item: item.tangency_id)),
        certification_tolerance=length_tolerance,
        angular_tolerance=angular_tolerance,
    )


def _derive_dandelin_section_plane_parts(
    construction: DandelinConstruction3D,
) -> _DandelinSectionPlaneParts:
    """Derive the true cutting-plane conic view without pseudo sphere circles."""

    if not isinstance(construction, DandelinConstruction3D):
        raise TypeError("construction must be a DandelinConstruction3D")
    length_tolerance, angular_tolerance = _view_tolerances(construction)
    diagram_id = f"{construction.construction_id}:view:section-plane"
    try:
        trace = compute_quadric_section(
            f"{diagram_id}:conic",
            construction.cone,
            construction.plane,
            context=construction.certification_context,
            coefficient_tolerance=construction.coefficient_tolerance,
        )
    except QuadricSectionError as exc:
        raise DandelinView2DError(
            f"section-plane conic cannot be certified: {exc}"
        ) from exc
    if trace.supporting_kind is not construction.supporting_kind:
        raise DandelinView2DError(
            "section-plane conic kind no longer matches the Dandelin construction"
        )

    frame = construction.section_frame
    foci = tuple(
        sorted(
            (
                DandelinDiagramPoint2D(
                    f"{diagram_id}:point:{record.focus_id}",
                    record.focus_id,
                    record.focus,
                )
                for record in construction.spheres
            ),
            key=lambda item: item.point_id,
        )
    )
    focus_map = {item.source_ref: item for item in foci}
    directrices = tuple(
        sorted(
            (
                DandelinDiagramLine2D(
                    f"{diagram_id}:line:{directrix.directrix_id}",
                    directrix.directrix_id,
                    directrix.point,
                    directrix.direction_coordinates,
                )
                for directrix in construction.directrices
            ),
            key=lambda item: item.line_id,
        )
    )
    sphere_plane_tangencies = tuple(
        sorted(
            (
                _sphere_plane_evidence(
                    construction,
                    record,
                    focus_map[record.focus_id],
                )
                for record in construction.spheres
            ),
            key=lambda item: item.tangency_id,
        )
    )
    return _DandelinSectionPlaneParts(
        diagram_id=diagram_id,
        frame=frame,
        supporting_kind=construction.supporting_kind,
        conic_trace=trace,
        focus_points=foci,
        directrices=directrices,
        sphere_plane_tangencies=sphere_plane_tangencies,
        certification_tolerance=length_tolerance,
        angular_tolerance=angular_tolerance,
    )


def build_dandelin_meridian_diagram(
    construction: DandelinConstruction3D,
) -> DandelinMeridianDiagram2D:
    """Derive the true cone-axis meridian diagram from one construction."""

    parts = _derive_dandelin_meridian_parts(construction)
    return DandelinMeridianDiagram2D(
        construction=construction,
        diagram_id=parts.diagram_id,
        frame=parts.frame,
        radial_source=parts.radial_source,
        section_line=parts.section_line,
        generators=parts.generators,
        sphere_circles=parts.sphere_circles,
        focus_points=parts.focus_points,
        tangencies=parts.tangencies,
        certification_tolerance=parts.certification_tolerance,
        angular_tolerance=parts.angular_tolerance,
    )


def build_dandelin_section_plane_diagram(
    construction: DandelinConstruction3D,
) -> DandelinSectionPlaneDiagram2D:
    """Derive the true cutting-plane conic view without pseudo sphere circles."""

    parts = _derive_dandelin_section_plane_parts(construction)
    return DandelinSectionPlaneDiagram2D(
        construction=construction,
        diagram_id=parts.diagram_id,
        frame=parts.frame,
        supporting_kind=parts.supporting_kind,
        conic_trace=parts.conic_trace,
        focus_points=parts.focus_points,
        directrices=parts.directrices,
        sphere_plane_tangencies=parts.sphere_plane_tangencies,
        certification_tolerance=parts.certification_tolerance,
        angular_tolerance=parts.angular_tolerance,
    )


def canonical_dandelin_meridian_diagram_json(
    diagram: DandelinMeridianDiagram2D,
) -> str:
    if not isinstance(diagram, DandelinMeridianDiagram2D):
        raise TypeError("diagram must be a DandelinMeridianDiagram2D")
    return diagram.canonical_json()


def canonical_dandelin_section_plane_diagram_json(
    diagram: DandelinSectionPlaneDiagram2D,
) -> str:
    if not isinstance(diagram, DandelinSectionPlaneDiagram2D):
        raise TypeError("diagram must be a DandelinSectionPlaneDiagram2D")
    return diagram.canonical_json()


__all__ = [
    "DANDELIN_MERIDIAN_DIAGRAM_2D_SCHEMA",
    "DANDELIN_SECTION_PLANE_DIAGRAM_2D_SCHEMA",
    "DandelinCircleLineTangency2D",
    "DandelinDiagramLine2D",
    "DandelinDiagramPoint2D",
    "DandelinDiagramSegment2D",
    "DandelinMeridianDiagram2D",
    "DandelinSectionPlaneDiagram2D",
    "DandelinSphereCircleSection2D",
    "DandelinSpherePlaneTangency2D",
    "DandelinView2DError",
    "build_dandelin_meridian_diagram",
    "build_dandelin_section_plane_diagram",
    "canonical_dandelin_meridian_diagram_json",
    "canonical_dandelin_section_plane_diagram_json",
]
