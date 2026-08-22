"""Contracts for renderer-neutral open-face unified compositing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json

import numpy as np

from ..compositor import CompositorCycleError, PainterConstraint, stable_topological_sort
from ..topology import ParameterInterval
from ..visibility import VisibilityKind
from .trace import OpenFaceVisibilityFrame


OPEN_FACE_UNIFIED_COMPOSITING_SCHEMA = (
    "manim-open-convex-face-unified-compositing/v1"
)


class OpenFaceUnifiedCompositingError(ValueError):
    """One exact deterministic painter graph could not be produced."""


class OpenFacePaintPolicy(str, Enum):
    DIAGRAMMATIC = "diagrammatic"
    PHYSICAL = "physical"

    @classmethod
    def parse(cls, value: "OpenFacePaintPolicy | str") -> "OpenFacePaintPolicy":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            raise OpenFaceUnifiedCompositingError(
                "paint_policy must be 'diagrammatic' or 'physical'"
            ) from exc


@dataclass(frozen=True, slots=True)
class OpenFaceUnifiedCompositingLimits:
    max_faces: int = 64
    max_paths: int = 128
    max_line_face_pairs: int = 4096
    max_line_line_pairs: int = 4096
    max_fragments_per_path: int = 1024
    max_total_fragments: int = 32768
    max_fragment_pair_candidates: int = 262144
    max_relations: int = 262144
    # Keep new limits after the original public positional fields.  Existing
    # callers may construct this exported dataclass positionally.
    max_fragment_face_candidates: int = 262144

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


OPEN_FACE_UNIFIED_COMPOSITING_LIMITS = OpenFaceUnifiedCompositingLimits()


@dataclass(frozen=True, slots=True)
class OpenFacePaintFace:
    item_id: str
    face_id: str
    logical_surface_id: str

    def __post_init__(self) -> None:
        if self.item_id != f"face:{self.face_id}":
            raise ValueError("paint-face item_id must be derived from face_id")
        if not self.face_id or not self.logical_surface_id:
            raise ValueError("paint-face identities must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "itemId": self.item_id,
            "faceId": self.face_id,
            "logicalSurfaceId": self.logical_surface_id,
        }


@dataclass(frozen=True, slots=True)
class PaintPathFragment:
    """One painter cell with a frame-local deterministic identity.

    ``fragment_id`` is stable only within one computed frame.  A later frame
    may add or remove painter events and therefore renumber fragments on the
    same source path.  Renderer bindings must use their own preallocated slot
    identity rather than treating this value as cross-frame lineage.
    """

    fragment_id: str
    source_path_id: str
    parameter_interval: ParameterInterval
    visibility_kind: VisibilityKind
    occluder_face_ids: tuple[str, ...] = ()
    occluder_logical_surface_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.fragment_id or not self.source_path_id:
            raise ValueError("path-fragment identities must be non-empty")
        if not isinstance(self.parameter_interval, ParameterInterval):
            raise TypeError("parameter_interval must be a ParameterInterval")
        if not isinstance(self.visibility_kind, VisibilityKind):
            raise TypeError("visibility_kind must be a VisibilityKind")
        if tuple(sorted(set(self.occluder_face_ids))) != self.occluder_face_ids:
            raise ValueError("occluder_face_ids must be sorted and unique")
        if (
            tuple(sorted(set(self.occluder_logical_surface_ids)))
            != self.occluder_logical_surface_ids
        ):
            raise ValueError(
                "occluder_logical_surface_ids must be sorted and unique"
            )
        if self.visibility_kind is VisibilityKind.VISIBLE and (
            self.occluder_face_ids or self.occluder_logical_surface_ids
        ):
            raise ValueError("visible fragments cannot carry occluders")
        if self.visibility_kind is VisibilityKind.HIDDEN and not self.occluder_face_ids:
            raise ValueError("hidden fragments require an occluder identity")

    @property
    def item_id(self) -> str:
        return self.fragment_id

    def to_dict(self) -> dict[str, object]:
        return {
            "fragmentId": self.fragment_id,
            "sourcePathId": self.source_path_id,
            "parameterInterval": {
                "start": self.parameter_interval.start,
                "end": self.parameter_interval.end,
            },
            "visibilityKind": self.visibility_kind.value,
            "occluderFaceIds": list(self.occluder_face_ids),
            "occluderLogicalSurfaceIds": list(
                self.occluder_logical_surface_ids
            ),
        }


@dataclass(frozen=True, slots=True)
class OpenFacePaintRelation:
    far_item_id: str
    near_item_id: str
    reason: str
    minimum_depth_difference: float
    maximum_depth_difference: float
    overlap_measure: float

    def __post_init__(self) -> None:
        if not self.far_item_id or not self.near_item_id or not self.reason:
            raise ValueError("paint relation identities and reason must be non-empty")
        if self.far_item_id == self.near_item_id:
            raise ValueError("paint relation cannot be a self relation")
        values = (
            self.minimum_depth_difference,
            self.maximum_depth_difference,
            self.overlap_measure,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("paint relation metrics must be finite")
        if self.minimum_depth_difference > self.maximum_depth_difference:
            raise ValueError("paint relation depth range is reversed")
        if self.overlap_measure < 0.0:
            raise ValueError("overlap_measure must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "farItemId": self.far_item_id,
            "nearItemId": self.near_item_id,
            "reason": self.reason,
            "minimumDepthDifference": self.minimum_depth_difference,
            "maximumDepthDifference": self.maximum_depth_difference,
            "overlapMeasure": self.overlap_measure,
        }


def _fragment_sort_key(
    fragment: PaintPathFragment,
) -> tuple[str, float, float, str]:
    interval = fragment.parameter_interval
    return (
        fragment.source_path_id,
        interval.start,
        interval.end,
        fragment.fragment_id,
    )


@dataclass(frozen=True, slots=True)
class OpenFaceUnifiedCompositingFrame:
    visibility: OpenFaceVisibilityFrame
    paint_policy: OpenFacePaintPolicy
    faces: tuple[OpenFacePaintFace, ...]
    path_fragments: tuple[PaintPathFragment, ...]
    order_relations: tuple[OpenFacePaintRelation, ...]
    draw_order: tuple[str, ...]
    schema: str = OPEN_FACE_UNIFIED_COMPOSITING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != OPEN_FACE_UNIFIED_COMPOSITING_SCHEMA:
            raise ValueError("invalid unified-compositing schema")
        if not isinstance(self.visibility, OpenFaceVisibilityFrame):
            raise TypeError("visibility must be an OpenFaceVisibilityFrame")
        if not isinstance(self.paint_policy, OpenFacePaintPolicy):
            raise TypeError("paint_policy must be an OpenFacePaintPolicy")
        if not all(isinstance(item, OpenFacePaintFace) for item in self.faces):
            raise TypeError("faces must contain OpenFacePaintFace values")

        expected_faces = tuple(
            (item.face_id, item.logical_surface_id)
            for item in self.visibility.face_tolerances
        )
        expected_face_ids = tuple(face_id for face_id, _ in expected_faces)
        if len(set(expected_face_ids)) != len(expected_face_ids):
            raise ValueError("visibility face identities must be unique")
        actual_faces = tuple(
            (item.face_id, item.logical_surface_id) for item in self.faces
        )
        if actual_faces != expected_faces:
            raise ValueError(
                "faces must exactly match visibility face identities in canonical order"
            )
        advisory_order = self.visibility.advisory_face_draw_order
        if (
            len(advisory_order) != len(expected_face_ids)
            or set(advisory_order) != set(expected_face_ids)
        ):
            raise ValueError(
                "visibility advisory face order must cover every face exactly once"
            )

        if not all(
            isinstance(item, PaintPathFragment) for item in self.path_fragments
        ):
            raise TypeError("path_fragments must contain PaintPathFragment values")
        canonical_fragments = tuple(
            sorted(self.path_fragments, key=_fragment_sort_key)
        )
        if self.path_fragments != canonical_fragments:
            raise ValueError(
                "path_fragments must use canonical source and parameter order"
            )

        visibility_edges = self.visibility.edges
        visibility_path_ids = tuple(item.source_edge_id for item in visibility_edges)
        if len(set(visibility_path_ids)) != len(visibility_path_ids):
            raise ValueError("visibility path identities must be unique")
        if visibility_path_ids != tuple(sorted(visibility_path_ids)):
            raise ValueError("visibility paths must use canonical identity order")

        fragments_by_path: dict[str, list[PaintPathFragment]] = {}
        for fragment in self.path_fragments:
            fragments_by_path.setdefault(fragment.source_path_id, []).append(fragment)
        unknown_paths = sorted(set(fragments_by_path) - set(visibility_path_ids))
        if unknown_paths:
            raise ValueError(
                "path fragments reference unknown visibility paths: "
                + ", ".join(unknown_paths)
            )
        missing_paths = sorted(set(visibility_path_ids) - set(fragments_by_path))
        if missing_paths:
            raise ValueError(
                "path fragments are missing visibility paths: "
                + ", ".join(missing_paths)
            )

        for edge in visibility_edges:
            fragments = fragments_by_path[edge.source_edge_id]
            epsilon = float(edge.parameter_epsilon)
            if not np.isfinite(epsilon) or epsilon < 0.0:
                raise ValueError(
                    f"visibility path {edge.source_edge_id!r} has invalid parameter epsilon"
                )
            if any(fragment.parameter_interval.length <= 0.0 for fragment in fragments):
                raise ValueError(
                    f"path {edge.source_edge_id!r} contains an empty painter fragment"
                )
            if abs(fragments[0].parameter_interval.start) > epsilon:
                raise ValueError(
                    f"path {edge.source_edge_id!r} fragments do not start at zero"
                )
            if abs(fragments[-1].parameter_interval.end - 1.0) > epsilon:
                raise ValueError(
                    f"path {edge.source_edge_id!r} fragments do not end at one"
                )
            for left, right in zip(fragments, fragments[1:]):
                delta = right.parameter_interval.start - left.parameter_interval.end
                if delta > epsilon:
                    raise ValueError(
                        f"path {edge.source_edge_id!r} fragment partition contains a gap"
                    )
                if delta < -epsilon:
                    raise ValueError(
                        f"path {edge.source_edge_id!r} fragment partition overlaps"
                    )

            for fragment in fragments:
                interval = fragment.parameter_interval
                matches = [
                    span
                    for span in edge.spans
                    if span.start - epsilon <= interval.start
                    and interval.end <= span.end + epsilon
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"path {edge.source_edge_id!r} fragment does not fit within "
                        "one visibility span"
                    )
                span = matches[0]
                try:
                    expected_kind = VisibilityKind(span.kind)
                except ValueError as exc:
                    raise ValueError(
                        f"path {edge.source_edge_id!r} has unsupported visibility "
                        f"kind {span.kind!r}"
                    ) from exc
                if fragment.visibility_kind is not expected_kind:
                    raise ValueError(
                        f"path {edge.source_edge_id!r} fragment visibility kind "
                        "disagrees with the visibility trace"
                    )
                if (
                    fragment.occluder_face_ids != span.occluder_face_ids
                    or fragment.occluder_logical_surface_ids
                    != span.occluder_logical_surface_ids
                ):
                    raise ValueError(
                        f"path {edge.source_edge_id!r} fragment occluders disagree "
                        "with the visibility trace"
                    )

        if not all(
            isinstance(item, OpenFacePaintRelation) for item in self.order_relations
        ):
            raise TypeError("order_relations must contain OpenFacePaintRelation values")
        canonical_relations = tuple(
            sorted(
                self.order_relations,
                key=lambda item: (item.far_item_id, item.near_item_id),
            )
        )
        if self.order_relations != canonical_relations:
            raise ValueError("order_relations must use canonical identity order")

        item_ids = self.item_ids
        item_set = set(item_ids)
        if len(item_set) != len(item_ids):
            raise ValueError("paint item identities must be unique")
        if set(self.draw_order) != item_set or len(self.draw_order) != len(item_ids):
            raise ValueError("draw_order must cover every paint item exactly once")

        directions: set[tuple[str, str]] = set()
        constraints: list[PainterConstraint[str]] = []
        for relation in self.order_relations:
            direction = (relation.far_item_id, relation.near_item_id)
            if relation.far_item_id not in item_set or relation.near_item_id not in item_set:
                raise ValueError("paint relation references an unknown item")
            if direction in directions:
                raise ValueError("duplicate paint relation direction")
            if (direction[1], direction[0]) in directions:
                raise ValueError("paint relations contain contradictory directions")
            directions.add(direction)
            constraints.append(PainterConstraint(*direction))

        try:
            canonical_order = stable_topological_sort(
                item_ids,
                constraints,
                key=lambda item_id: item_id,
            )
        except CompositorCycleError as exc:
            unresolved = ", ".join(sorted(str(item) for item in exc.unresolved))
            raise ValueError(
                "paint relations contain a cycle: " + unresolved
            ) from exc

        rank = {item_id: index for index, item_id in enumerate(self.draw_order)}
        for relation in self.order_relations:
            if rank[relation.far_item_id] >= rank[relation.near_item_id]:
                raise ValueError(
                    "draw_order contradicts a paint relation: "
                    f"{relation.far_item_id!r} must precede "
                    f"{relation.near_item_id!r}"
                )
        if self.draw_order != canonical_order:
            raise ValueError(
                "draw_order must be the canonical deterministic topological order"
            )

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(
            (
                *(item.item_id for item in self.faces),
                *(item.fragment_id for item in self.path_fragments),
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "paintPolicy": self.paint_policy.value,
            "visibility": self.visibility.to_dict(),
            "faces": [item.to_dict() for item in self.faces],
            "pathFragments": [item.to_dict() for item in self.path_fragments],
            "orderRelations": [item.to_dict() for item in self.order_relations],
            "drawOrder": list(self.draw_order),
        }


def canonical_open_face_unified_compositing_json(
    frame: OpenFaceUnifiedCompositingFrame,
) -> str:
    return json.dumps(
        frame.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "OPEN_FACE_UNIFIED_COMPOSITING_LIMITS",
    "OPEN_FACE_UNIFIED_COMPOSITING_SCHEMA",
    "OpenFacePaintFace",
    "OpenFacePaintPolicy",
    "OpenFacePaintRelation",
    "OpenFaceUnifiedCompositingError",
    "OpenFaceUnifiedCompositingFrame",
    "OpenFaceUnifiedCompositingLimits",
    "PaintPathFragment",
    "canonical_open_face_unified_compositing_json",
]
