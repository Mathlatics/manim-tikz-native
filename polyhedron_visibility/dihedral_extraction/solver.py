from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..contract import ContractError, TolerancePolicy
from ..parallel_solver import (
    ParallelView,
    SolverError,
    _segment_face_interval_result,
    _spans_from_intervals,
)
from ..trace import (
    EdgeVisibility,
    FaceToleranceTrace,
    RawOcclusionInterval,
    SkippedFace,
    VisibilityFrame,
)
from .contract import (
    DerivedDihedralContractError,
    DerivedDihedralModel,
    RigidTransform3D,
)
from .trace import DerivedDihedralVisibilityFrame


class DerivedDihedralSolverError(ValueError):
    """Raised when one world frame cannot be solved without guessing."""


def _positions(
    model: DerivedDihedralModel,
    solid_positions: Mapping[str, Sequence[float]] | None,
    transform: RigidTransform3D,
    policy: TolerancePolicy,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw = model.solid.entry_positions if solid_positions is None else solid_positions
    try:
        model.solid.validate(
            vertex_positions=raw,
            require_closed_convex_manifold=True,
            tolerance_policy=policy,
        )
    except ContractError as exc:
        raise DerivedDihedralSolverError(f"invalid source solid frame: {exc}") from exc
    solid = {key: np.asarray(raw[key], dtype=float) for key in sorted(raw)}
    extracted = {
        key: transform.apply(model.solid.vertex_map[key].entry_position)
        for key in model.extracted_vertex_ids
    }
    return solid, extracted


def _cycles_match(
    first: Sequence[np.ndarray],
    second: Sequence[np.ndarray],
    tolerance: float,
) -> bool:
    if len(first) != len(second):
        return False
    actual = np.asarray(first, dtype=float)
    expected = np.asarray(second, dtype=float)
    for candidate in (expected, expected[::-1]):
        for offset in range(len(candidate)):
            if float(
                np.max(
                    np.linalg.norm(actual - np.roll(candidate, -offset, axis=0), axis=1)
                )
            ) <= tolerance:
                return True
    return False


def compute_derived_dihedral_visibility(
    model: DerivedDihedralModel,
    *,
    transform: RigidTransform3D,
    projection_matrix: Sequence[Sequence[float]],
    solid_vertex_positions: Mapping[str, Sequence[float]] | None = None,
    tolerance_policy: TolerancePolicy | None = None,
) -> DerivedDihedralVisibilityFrame:
    if not isinstance(model, DerivedDihedralModel):
        raise DerivedDihedralSolverError("model must be a DerivedDihedralModel")
    if not isinstance(transform, RigidTransform3D):
        raise DerivedDihedralSolverError("transform provider must return RigidTransform3D")
    policy = tolerance_policy or TolerancePolicy()
    try:
        view = ParallelView.from_matrix(projection_matrix)
        solid_positions, extracted_positions = _positions(
            model, solid_vertex_positions, transform, policy
        )
    except (SolverError, DerivedDihedralContractError) as exc:
        raise DerivedDihedralSolverError(str(exc)) from exc

    world_positions = {
        **{
            model.solid_vertex_id(key): value
            for key, value in solid_positions.items()
        },
        **{
            model.extracted_vertex_id(key): value
            for key, value in extracted_positions.items()
        },
    }
    overlay_model = model.overlay_model()
    coincident: list[str] = []
    for source_face_id in model.extraction.source_face_ids:
        source = model.solid.face_map[source_face_id]
        source_points = [solid_positions[item] for item in source.vertex_ids]
        extracted_points = [extracted_positions[item] for item in source.vertex_ids]
        tolerance = policy.resolve((*source_points, *extracted_points)).boundary
        if _cycles_match(source_points, extracted_points, tolerance):
            coincident.append(source_face_id)

    suppressed: list[str] = []
    coincident_set = set(coincident)
    for boundary in model.extraction.boundary_strokes:
        if not set(boundary.incident_source_face_ids) & coincident_set:
            continue
        start, end = boundary.vertex_ids
        tolerance = policy.resolve(
            (
                solid_positions[start],
                solid_positions[end],
                extracted_positions[start],
                extracted_positions[end],
            ),
            edge_length=float(np.linalg.norm(solid_positions[end] - solid_positions[start])),
        ).boundary
        if (
            float(np.linalg.norm(solid_positions[start] - extracted_positions[start]))
            <= tolerance
            and float(np.linalg.norm(solid_positions[end] - extracted_positions[end]))
            <= tolerance
        ):
            suppressed.append(model.solid_stroke_id(boundary.source_stroke_id))

    face_tolerances = tuple(
        FaceToleranceTrace(
            face.face_id,
            (resolved := policy.resolve(
                [world_positions[item] for item in face.vertex_ids]
            )).world,
            resolved.boundary,
            resolved.depth,
            resolved.angular,
        )
        for face in sorted(overlay_model.faces, key=lambda item: item.face_id)
    )
    all_surface_points = [
        world_positions[item]
        for face in overlay_model.faces
        for item in face.vertex_ids
    ]
    tolerance = policy.resolve(all_surface_points)
    face_draw_order = tuple(
        face_id
        for _depth, face_id in sorted(
            (
                float(
                    np.dot(
                        np.mean(
                            [world_positions[item] for item in face.vertex_ids], axis=0
                        ),
                        view.view_direction,
                    )
                ),
                face.face_id,
            )
            for face in overlay_model.faces
        )
    )

    edges: list[EdgeVisibility] = []
    for stroke in overlay_model.strokes:
        start = world_positions[stroke.vertex_ids[0]]
        end = world_positions[stroke.vertex_ids[1]]
        length = float(np.linalg.norm(end - start))
        edge_tolerance = policy.resolve((start, end), edge_length=length)
        if length <= edge_tolerance.world:
            raise DerivedDihedralSolverError(
                f"semantic stroke {stroke.source_edge_id} has zero length"
            )
        raw_intervals: list[RawOcclusionInterval] = []
        skipped: list[SkippedFace] = []
        if stroke.visibility_mode == "always_visible":
            skipped.extend(
                SkippedFace(face.face_id, "stroke_always_visible")
                for face in overlay_model.faces
            )
        elif stroke.visibility_mode == "always_hidden":
            raw_intervals.append(RawOcclusionInterval("__policy__", 0.0, 1.0))
        else:
            is_solid_stroke = stroke.source_edge_id.startswith("solid:")
            source_stroke = (
                model.solid.stroke_map.get(stroke.source_edge_id.removeprefix("solid:"))
                if is_solid_stroke
                else None
            )
            for face in overlay_model.faces:
                if face.face_id in stroke.incident_face_ids:
                    skipped.append(SkippedFace(face.face_id, "incident_face"))
                    continue
                if not face.occludes_strokes:
                    skipped.append(SkippedFace(face.face_id, "occlusion_disabled"))
                    continue
                if is_solid_stroke and face.face_id.startswith(
                    model.extraction.entity_id + ":"
                ):
                    source_face_id = face.face_id.split(":", 1)[1]
                    if (
                        source_face_id in coincident_set
                        and source_stroke is not None
                        and source_face_id in source_stroke.incident_face_ids
                    ):
                        skipped.append(
                            SkippedFace(face.face_id, "coincident_clone_incident")
                        )
                        continue
                if (
                    not is_solid_stroke
                    and face.face_id.startswith("solid:")
                    and face.face_id.removeprefix("solid:") in coincident_set
                ):
                    source_face_id = face.face_id.removeprefix("solid:")
                    derived_incidents = {
                        item.removeprefix(model.extraction.entity_id + ":")
                        for item in stroke.incident_face_ids
                    }
                    if source_face_id in derived_incidents:
                        skipped.append(
                            SkippedFace(face.face_id, "coincident_source_incident")
                        )
                        continue
                result = _segment_face_interval_result(
                    start,
                    end,
                    [world_positions[item] for item in face.vertex_ids],
                    view,
                    tolerance_policy=policy,
                )
                if result.interval is None:
                    skipped.append(
                        SkippedFace(face.face_id, result.reason or "no_occlusion")
                    )
                else:
                    raw_intervals.append(
                        RawOcclusionInterval(
                            face.face_id,
                            result.interval[0],
                            result.interval[1],
                        )
                    )
        raw_intervals.sort(key=lambda item: (item.start, item.end, item.face_id))
        skipped.sort(key=lambda item: (item.face_id, item.reason))
        edges.append(
            EdgeVisibility(
                stroke.source_edge_id,
                tuple(raw_intervals),
                tuple(skipped),
                _spans_from_intervals(raw_intervals, edge_tolerance.parameter),
                edge_tolerance.parameter,
                face_tolerances,
            )
        )

    line_frame = VisibilityFrame(
        model.visibility_group_id,
        view.projection_matrix,
        view.view_direction,
        tolerance,
        tuple(sorted(edges, key=lambda item: item.source_edge_id)),
        face_draw_order,
    )
    return DerivedDihedralVisibilityFrame(
        line_frame,
        transform,
        tuple(sorted(coincident)),
        tuple(sorted(suppressed)),
    )


__all__ = [
    "DerivedDihedralSolverError",
    "compute_derived_dihedral_visibility",
]
