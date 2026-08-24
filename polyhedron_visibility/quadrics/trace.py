"""Immutable renderer-neutral traces for plane/quadric sections."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Sequence

import numpy as np

from ..topology import ParameterInterval
from .conics import ConicKind, ConicParameterization
from .curves import ParametricConicBranch


QUADRIC_SECTION_TRACE_SCHEMA = "manim-quadric-section-trace/v1"


class FiniteSectionTopology(str, Enum):
    """Topology after a supporting conic is clipped to a finite surface."""

    EMPTY = "empty"
    POINT = "point"
    MULTIPLE_POINTS = "multiple_points"
    CLOSED_CURVE = "closed_curve"
    OPEN_CURVE = "open_curve"
    MULTIPLE_OPEN_CURVES = "multiple_open_curves"
    CURVES_AND_POINTS = "curves_and_points"


def _matrix(value: Sequence[Sequence[float]], shape: tuple[int, int], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite {shape[0]}x{shape[1]} matrix")
    return result


def _point(value: Sequence[float], size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain {size} finite coordinates")
    return result


@dataclass(frozen=True, slots=True)
class SectionBranchTrace:
    """One stable semantic branch and its affine plane-to-world embedding."""

    branch_id: str
    parameterization: ConicParameterization
    plane_embedding: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    def __post_init__(self) -> None:
        if not isinstance(self.branch_id, str) or not self.branch_id:
            raise ValueError("branch_id must be a non-empty string")
        if not isinstance(self.parameterization, ConicParameterization):
            raise TypeError("parameterization must be a ConicParameterization")
        embedding = _matrix(self.plane_embedding, (4, 3), "plane_embedding")
        if not np.allclose(
            embedding[3], np.asarray((0.0, 0.0, 1.0)), rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("plane_embedding must use affine homogeneous coordinates")

    def plane_point(self, parameter: float) -> np.ndarray:
        return self.parameterization.point(parameter)

    def world_point(self, parameter: float) -> np.ndarray:
        uv = self.plane_point(parameter)
        homogeneous = np.asarray((uv[0], uv[1], 1.0), dtype=float)
        world_homogeneous = np.asarray(self.plane_embedding, dtype=float) @ homogeneous
        if abs(float(world_homogeneous[3]) - 1.0) > 1.0e-10:
            raise ValueError("plane embedding produced a non-affine world point")
        return world_homogeneous[:3].copy()

    def world_tangent(self, parameter: float) -> np.ndarray:
        tangent = self.parameterization.tangent(parameter)
        embedding = np.asarray(self.plane_embedding, dtype=float)
        return embedding[:3, :2] @ tangent

    def to_dict(self) -> dict[str, object]:
        natural = self.parameterization.natural_domain
        return {
            "branchId": self.branch_id,
            "kind": self.parameterization.kind.value,
            "branchLabel": self.parameterization.branch_label,
            "origin": list(self.parameterization.origin),
            "firstAxis": list(self.parameterization.first_axis),
            "secondAxis": list(self.parameterization.second_axis),
            "branchSign": self.parameterization.branch_sign,
            "naturalDomain": (
                None if natural is None else [natural.start, natural.end]
            ),
            "closed": self.parameterization.closed,
            "planeEmbedding": [list(row) for row in self.plane_embedding],
        }


@dataclass(frozen=True, slots=True)
class SectionComponentTrace:
    """One connected finite component, possibly wrapping a periodic seam."""

    component_id: str
    branch_id: str
    parameter_intervals: tuple[ParameterInterval, ...]
    closed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.component_id, str) or not self.component_id:
            raise ValueError("component_id must be a non-empty string")
        if not isinstance(self.branch_id, str) or not self.branch_id:
            raise ValueError("branch_id must be a non-empty string")
        if not self.parameter_intervals:
            raise ValueError("a section component requires a parameter interval")
        if not all(
            isinstance(item, ParameterInterval) and item.length > 0.0
            for item in self.parameter_intervals
        ):
            raise ValueError("section parameter intervals must have positive length")
        ordered = tuple(
            sorted(self.parameter_intervals, key=lambda item: (item.start, item.end))
        )
        if ordered != self.parameter_intervals:
            raise ValueError("section parameter intervals must use canonical order")
        for left, right in zip(ordered, ordered[1:]):
            if right.start < left.end:
                raise ValueError("section parameter intervals must not overlap")

    @property
    def parameter_start(self) -> float:
        return self.parameter_intervals[0].start

    @property
    def parameter_end(self) -> float:
        return self.parameter_intervals[-1].end

    def to_dict(self) -> dict[str, object]:
        return {
            "componentId": self.component_id,
            "branchId": self.branch_id,
            "parameterIntervals": [
                {"start": interval.start, "end": interval.end}
                for interval in self.parameter_intervals
            ],
            "closed": self.closed,
        }


@dataclass(frozen=True, slots=True)
class QuadricSectionTrace:
    """One exact supporting conic plus its finite-surface clipping result."""

    section_id: str
    surface_id: str
    supporting_kind: ConicKind
    finite_topology: FiniteSectionTopology
    conic_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    plane_embedding: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    branches: tuple[SectionBranchTrace, ...] = ()
    components: tuple[SectionComponentTrace, ...] = ()
    isolated_world_points: tuple[tuple[float, float, float], ...] = ()
    schema: str = QUADRIC_SECTION_TRACE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != QUADRIC_SECTION_TRACE_SCHEMA:
            raise ValueError("invalid quadric-section trace schema")
        if not isinstance(self.section_id, str) or not self.section_id:
            raise ValueError("section_id must be a non-empty string")
        if not isinstance(self.surface_id, str) or not self.surface_id:
            raise ValueError("surface_id must be a non-empty string")
        if not isinstance(self.supporting_kind, ConicKind):
            raise TypeError("supporting_kind must be a ConicKind")
        if not isinstance(self.finite_topology, FiniteSectionTopology):
            raise TypeError("finite_topology must be a FiniteSectionTopology")
        _matrix(self.conic_matrix, (3, 3), "conic_matrix")
        _matrix(self.plane_embedding, (4, 3), "plane_embedding")
        branch_ids = tuple(item.branch_id for item in self.branches)
        if len(set(branch_ids)) != len(branch_ids):
            raise ValueError("section branch identities must be unique")
        if branch_ids != tuple(sorted(branch_ids)):
            raise ValueError("section branches must use canonical identity order")
        component_ids = tuple(item.component_id for item in self.components)
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("section component identities must be unique")
        if component_ids != tuple(sorted(component_ids)):
            raise ValueError("section components must use canonical identity order")
        unknown = sorted(
            {item.branch_id for item in self.components} - set(branch_ids)
        )
        if unknown:
            raise ValueError(
                "section components reference unknown branches: " + ", ".join(unknown)
            )
        for point in self.isolated_world_points:
            _point(point, 3, "isolated_world_point")

        component_count = len(self.components)
        point_count = len(self.isolated_world_points)
        if self.finite_topology is FiniteSectionTopology.EMPTY:
            if component_count or point_count:
                raise ValueError("an empty finite section cannot contain geometry")
        elif self.finite_topology is FiniteSectionTopology.POINT:
            if point_count != 1 or component_count:
                raise ValueError("a point section must contain exactly one point")
        elif self.finite_topology is FiniteSectionTopology.MULTIPLE_POINTS:
            if point_count < 2 or component_count:
                raise ValueError(
                    "a multiple-point section must contain at least two points"
                )
        elif self.finite_topology is FiniteSectionTopology.CLOSED_CURVE:
            if component_count != 1 or not self.components[0].closed or point_count:
                raise ValueError("a closed section must contain one closed component")
        elif self.finite_topology is FiniteSectionTopology.OPEN_CURVE:
            if component_count != 1 or self.components[0].closed or point_count:
                raise ValueError("an open section must contain one open component")
        elif self.finite_topology is FiniteSectionTopology.MULTIPLE_OPEN_CURVES:
            if component_count < 2 or any(item.closed for item in self.components) or point_count:
                raise ValueError(
                    "a multiple-open section requires at least two open components"
                )
        elif self.finite_topology is FiniteSectionTopology.CURVES_AND_POINTS:
            if not component_count or not point_count:
                raise ValueError(
                    "a mixed section requires both curves and isolated points"
                )

    @property
    def branch_map(self) -> dict[str, SectionBranchTrace]:
        return {item.branch_id: item for item in self.branches}

    @property
    def component_map(self) -> dict[str, SectionComponentTrace]:
        return {item.component_id: item for item in self.components}

    def world_point(self, component_id: str, parameter: float) -> np.ndarray:
        component = self.component_map[component_id]
        if not any(interval.contains(parameter) for interval in component.parameter_intervals):
            raise ValueError("parameter lies outside the finite section component")
        return self.branch_map[component.branch_id].world_point(parameter)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sectionId": self.section_id,
            "surfaceId": self.surface_id,
            "supportingKind": self.supporting_kind.value,
            "finiteTopology": self.finite_topology.value,
            "conicMatrix": [list(row) for row in self.conic_matrix],
            "planeEmbedding": [list(row) for row in self.plane_embedding],
            "branches": [branch.to_dict() for branch in self.branches],
            "components": [component.to_dict() for component in self.components],
            "isolatedWorldPoints": [list(point) for point in self.isolated_world_points],
        }


def canonical_quadric_section_trace_json(frame: QuadricSectionTrace) -> str:
    return json.dumps(
        frame.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def section_trace_curves(
    trace: QuadricSectionTrace,
) -> tuple[ParametricConicBranch, ...]:
    """Adapt every finite curve interval into the visibility curve contract.

    Isolated section points deliberately remain in
    ``trace.isolated_world_points``.  A periodic component split across its
    parameter seam becomes two curve records because each analytic curve owns
    one finite contiguous domain.
    """

    if not isinstance(trace, QuadricSectionTrace):
        raise TypeError("trace must be a QuadricSectionTrace")
    branches = trace.branch_map
    result: list[ParametricConicBranch] = []
    for component in trace.components:
        branch = branches[component.branch_id]
        for index, interval in enumerate(component.parameter_intervals):
            curve_id = (
                component.component_id
                if len(component.parameter_intervals) == 1
                else f"{component.component_id}:interval:{index:04d}"
            )
            result.append(
                ParametricConicBranch(
                    curve_id,
                    branch.parameterization,
                    branch.plane_embedding,
                    interval,
                )
            )
    return tuple(result)


__all__ = [
    "FiniteSectionTopology",
    "QUADRIC_SECTION_TRACE_SCHEMA",
    "QuadricSectionTrace",
    "SectionBranchTrace",
    "SectionComponentTrace",
    "canonical_quadric_section_trace_json",
    "section_trace_curves",
]
