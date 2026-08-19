from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from manim import ManimColor, Mobject, Polygon, VGroup
from manim import Line

from ..api import ParallelProjection
from ..authoring import OcclusionScene3D, VertexPositionProvider
from ..binding import DisplayPointProvider
from ..contract import TolerancePolicy, VisibilityModel
from ..style import OcclusionStyle
from .contract import (
    DerivedDihedralContractError,
    DerivedDihedralModel,
    RigidTransform3D,
)
from .base_plane import BasePlaneRotation3D


class DerivedDihedralAuthoringError(ValueError):
    """Raised before a derived dihedral mutates a Manim Scene."""


TransformProvider = Callable[[], RigidTransform3D]


@dataclass
class ExtractedDihedralEntity3D:
    model: DerivedDihedralModel
    transform_provider: TransformProvider
    stroke_mobjects: Mapping[str, Line]
    face_mobjects: Mapping[str, Polygon]
    mobject: VGroup

    def current_transform(self) -> RigidTransform3D:
        value = self.transform_provider()
        if not isinstance(value, RigidTransform3D):
            raise DerivedDihedralAuthoringError(
                "transform_provider must return RigidTransform3D"
            )
        return value

    def current_positions(self) -> dict[str, np.ndarray]:
        return self.positions_for_transform(self.current_transform())

    def positions_for_transform(
        self,
        transform: RigidTransform3D,
    ) -> dict[str, np.ndarray]:
        if not isinstance(transform, RigidTransform3D):
            raise DerivedDihedralAuthoringError(
                "transform must be a RigidTransform3D"
            )
        return {
            vertex_id: transform.apply(
                self.model.solid.vertex_map[vertex_id].entry_position
            )
            for vertex_id in self.model.extracted_vertex_ids
        }

    def update_mobjects(
        self,
        display_point_provider: DisplayPointProvider | None = None,
        *,
        positions: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        current = (
            self.current_positions()
            if positions is None
            else {
                vertex_id: np.asarray(positions[vertex_id], dtype=float)
                for vertex_id in self.model.extracted_vertex_ids
            }
        )

        def display(point: Sequence[float]) -> np.ndarray:
            value = (
                point
                if display_point_provider is None
                else display_point_provider(point)
            )
            result = np.asarray(value, dtype=float)
            if result.shape != (3,) or not np.all(np.isfinite(result)):
                raise DerivedDihedralAuthoringError(
                    "display_point_provider must return a finite three-component point"
                )
            return result

        for boundary in self.model.extraction.boundary_strokes:
            source = self.stroke_mobjects[
                self.model.extracted_stroke_id(boundary.source_stroke_id)
            ]
            source.put_start_and_end_on(
                display(current[boundary.vertex_ids[0]]),
                display(current[boundary.vertex_ids[1]]),
            )
        for face_id in self.model.extraction.source_face_ids:
            source_face = self.model.solid.face_map[face_id]
            polygon = self.face_mobjects[self.model.extracted_face_id(face_id)]
            points = [display(current[item]) for item in source_face.vertex_ids]
            polygon.set_points_as_corners([*points, points[0]])


class ExtractedDihedralScene3D:
    """Author one closed solid and extract one independently movable dihedral."""

    def __init__(self, visibility_group_id: str) -> None:
        self._solid = OcclusionScene3D(
            visibility_group_id,
            topology_mode="closed_convex_polyhedron",
        )
        self._face_fill_bindings: dict[str, Polygon] = {}
        self._entity: ExtractedDihedralEntity3D | None = None

    @property
    def frozen(self) -> bool:
        return self._solid.frozen

    def vertex(
        self,
        vertex_id: str,
        position_provider: VertexPositionProvider,
    ) -> "ExtractedDihedralScene3D":
        self._solid.vertex(vertex_id, position_provider)
        return self

    def face(
        self,
        face_id: str,
        vertex_ids: Sequence[str],
        *,
        occludes_strokes: bool = True,
        source_mobject: Polygon | None = None,
    ) -> "ExtractedDihedralScene3D":
        self._solid.face(
            face_id,
            vertex_ids,
            occludes_strokes=occludes_strokes,
        )
        if source_mobject is not None:
            if not isinstance(source_mobject, Polygon):
                raise DerivedDihedralAuthoringError(
                    f"face {face_id} source_mobject must be one native Manim Polygon"
                )
            self._face_fill_bindings[face_id.strip()] = source_mobject
        return self

    def stroke(
        self,
        source_edge_id: str,
        start_vertex_id: str,
        end_vertex_id: str,
        source_mobject: Mobject,
        *,
        incident_face_ids: Sequence[str] | None = None,
        visibility_mode: str = "auto",
    ) -> "ExtractedDihedralScene3D":
        self._solid.stroke(
            source_edge_id,
            start_vertex_id,
            end_vertex_id,
            source_mobject,
            incident_face_ids=incident_face_ids,
            visibility_mode=visibility_mode,
        )
        return self

    def current_solid_positions(self) -> dict[str, Sequence[float]]:
        return self._solid.current_positions()

    def freeze(
        self,
        *,
        tolerance_policy: TolerancePolicy | None = None,
    ) -> VisibilityModel:
        return self._solid.freeze(tolerance_policy=tolerance_policy)

    def base_plane_rotation(
        self,
        face_id: str,
        *,
        target_outward_normal: Sequence[float] = (0.0, 0.0, -1.0),
        anchor: Sequence[float] | None = None,
        tolerance_policy: TolerancePolicy | None = None,
    ) -> BasePlaneRotation3D:
        """Freeze a motion that makes one solid face the horizontal base."""

        model = self.freeze(tolerance_policy=tolerance_policy)
        return BasePlaneRotation3D.from_model(
            model,
            face_id,
            vertex_positions=self.current_solid_positions(),
            target_outward_normal=target_outward_normal,
            anchor=anchor,
            tolerance_policy=tolerance_policy,
        )

    def extract_dihedral(
        self,
        entity_id: str,
        source_face_ids: Sequence[str],
        *,
        transform_provider: TransformProvider,
        edge_color: object | None = None,
        face_color: object | None = None,
        face_opacity: float | None = None,
        tolerance_policy: TolerancePolicy | None = None,
    ) -> ExtractedDihedralEntity3D:
        if self._entity is not None:
            raise DerivedDihedralAuthoringError(
                "derived-dihedral v1 accepts exactly one extracted entity"
            )
        if not callable(transform_provider):
            raise DerivedDihedralAuthoringError(
                "transform_provider must be callable"
            )
        try:
            entry_transform = transform_provider()
        except Exception as exc:
            raise DerivedDihedralAuthoringError(
                "transform_provider failed while freezing the entry transform"
            ) from exc
        if not isinstance(entry_transform, RigidTransform3D):
            raise DerivedDihedralAuthoringError(
                "transform_provider must return RigidTransform3D"
            )
        solid = self.freeze(tolerance_policy=tolerance_policy)
        try:
            model = DerivedDihedralModel.from_solid(
                solid.visibility_group_id,
                solid,
                entity_id=entity_id,
                source_face_ids=source_face_ids,
                entry_transform=entry_transform,
                vertex_positions=self.current_solid_positions(),
                tolerance_policy=tolerance_policy,
            )
        except DerivedDihedralContractError as exc:
            raise DerivedDihedralAuthoringError(str(exc)) from exc

        missing_strokes = sorted(
            {
                item.source_stroke_id
                for item in model.extraction.boundary_strokes
            }
            - set(self._solid.stroke_bindings)
        )
        if missing_strokes:
            raise DerivedDihedralAuthoringError(
                "missing source stroke bindings: " + ", ".join(missing_strokes)
            )
        positions = {
            key: entry_transform.apply(solid.vertex_map[key].entry_position)
            for key in model.extracted_vertex_ids
        }
        source_stroke_z = [
            float(getattr(item, "z_index", 0.0))
            for item in self._solid.stroke_bindings.values()
        ]
        derived_stroke_z = max(source_stroke_z, default=0.0) + 1.0
        stroke_mobjects: dict[str, Line] = {}
        for boundary_index, boundary in enumerate(model.extraction.boundary_strokes):
            original = self._solid.stroke_bindings[boundary.source_stroke_id]
            if not isinstance(original, Line) or tuple(original.get_family()) != (original,):
                raise DerivedDihedralAuthoringError(
                    f"source stroke {boundary.source_stroke_id} must be one complete Manim Line"
                )
            duplicate = original.copy()
            duplicate.put_start_and_end_on(
                positions[boundary.vertex_ids[0]],
                positions[boundary.vertex_ids[1]],
            )
            if edge_color is not None:
                duplicate.set_color(edge_color)
            duplicate.set_z_index(derived_stroke_z + boundary_index)
            stroke_mobjects[
                model.extracted_stroke_id(boundary.source_stroke_id)
            ] = duplicate

        face_mobjects: dict[str, Polygon] = {}
        selected_bound_faces = set(model.extraction.source_face_ids) & set(
            self._face_fill_bindings
        )
        if selected_bound_faces and selected_bound_faces != set(
            model.extraction.source_face_ids
        ):
            raise DerivedDihedralAuthoringError(
                "extracted face fill binding must be present for both selected faces or neither"
            )
        source_face_z = [
            float(getattr(item, "z_index", 0.0))
            for item in self._face_fill_bindings.values()
        ]
        derived_face_z = max(source_face_z, default=0.0) + 1.0
        for face_index, face_id in enumerate(model.extraction.source_face_ids):
            source_face = solid.face_map[face_id]
            if face_id in self._face_fill_bindings:
                duplicate = self._face_fill_bindings[face_id].copy()
                if face_color is not None:
                    duplicate.set_fill(color=face_color)
                if face_opacity is not None:
                    if not isinstance(face_opacity, (int, float)) or not 0 <= float(face_opacity) <= 1:
                        raise DerivedDihedralAuthoringError(
                            "face_opacity must be between 0 and 1"
                        )
                    duplicate.set_fill(opacity=float(face_opacity))
            else:
                duplicate = Polygon(
                    *(positions[item] for item in source_face.vertex_ids),
                    fill_opacity=float(face_opacity or 0.0),
                    stroke_opacity=0.0,
                )
                if face_color is not None:
                    duplicate.set_fill(color=face_color)
            points = [positions[item] for item in source_face.vertex_ids]
            duplicate.set_points_as_corners([*points, points[0]])
            duplicate.set_z_index(derived_face_z + face_index)
            face_mobjects[model.extracted_face_id(face_id)] = duplicate

        group = VGroup(
            *(face_mobjects[key] for key in sorted(face_mobjects)),
            *(stroke_mobjects[key] for key in sorted(stroke_mobjects)),
        )
        self._entity = ExtractedDihedralEntity3D(
            model,
            transform_provider,
            stroke_mobjects,
            face_mobjects,
            group,
        )
        return self._entity

    @property
    def entity(self) -> ExtractedDihedralEntity3D:
        if self._entity is None:
            raise DerivedDihedralAuthoringError(
                "call extract_dihedral before requesting the entity or controller"
            )
        return self._entity

    @property
    def solid_stroke_bindings(self) -> Mapping[str, Mobject]:
        return self._solid.stroke_bindings

    @property
    def solid_face_fill_bindings(self) -> Mapping[str, Polygon]:
        return dict(self._face_fill_bindings)

    def controller(
        self,
        scene: object,
        *,
        projection: ParallelProjection,
        style: OcclusionStyle,
        display_point_provider: DisplayPointProvider | None = None,
        tolerance_policy: TolerancePolicy | None = None,
        source_coordinate_mode: str = "world",
        accurate_transparency: bool = False,
        unified_compositing: bool | None = None,
        unified_fragment_slots_per_style: int = 12,
        global_transform_provider: TransformProvider | None = None,
        identity_handoff_distance: float = 0.12,
    ) -> object:
        from .manim import ExtractedDihedralOcclusion3D

        entity = self.entity
        stroke_bindings = {
            **{
                entity.model.solid_stroke_id(key): value
                for key, value in self.solid_stroke_bindings.items()
            },
            **dict(entity.stroke_mobjects),
        }
        face_bindings = {
            **{
                entity.model.solid_face_id(key): value
                for key, value in self.solid_face_fill_bindings.items()
            },
            **dict(entity.face_mobjects),
        }
        return ExtractedDihedralOcclusion3D(
            scene,
            entity.model,
            entity=entity,
            solid_position_provider=self.current_solid_positions,
            stroke_bindings=stroke_bindings,
            face_fill_bindings=face_bindings or None,
            projection=projection,
            display_point_provider=display_point_provider,
            style=style,
            tolerance_policy=tolerance_policy,
            source_coordinate_mode=source_coordinate_mode,
            accurate_transparency=accurate_transparency,
            unified_compositing=unified_compositing,
            unified_fragment_slots_per_style=(
                unified_fragment_slots_per_style
            ),
            global_transform_provider=global_transform_provider,
            identity_handoff_distance=identity_handoff_distance,
        )


__all__ = [
    "DerivedDihedralAuthoringError",
    "ExtractedDihedralEntity3D",
    "ExtractedDihedralScene3D",
    "TransformProvider",
]
