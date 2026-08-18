from __future__ import annotations

from typing import Callable, Mapping, Sequence

from manim import Mobject

from ..api import ParallelProjection
from ..authoring import OcclusionAuthoringError, OcclusionScene3D, VertexPositionProvider
from ..binding import DisplayPointProvider
from ..contract import TolerancePolicy, VisibilityModel
from ..depth_cue import FaceDepthCueStyle
from ..style import OcclusionStyle
from .contract import SectionPlane3D
from .manim import ConvexSection3D, ConvexSectionStyle, PlaneProvider
from .solver import (
    intersect_plane_with_convex_polyhedron,
    intersect_segment_with_convex_polyhedron,
)
from .trace import ConvexSectionFrame, SegmentSolidIntersection


class ConvexSectionAuthoringError(OcclusionAuthoringError):
    """Raised while declaring one closed solid, free lines, and one section."""


class ConvexSectionScene3D:
    """Fluent authoring API for a convex solid cut by one infinite plane.

    This builder deliberately reuses the closed-solid topology contract.  A
    free semantic line is simply a registered stroke whose endpoints do not
    both belong to a solid face; it automatically participates in every solid
    face and in the automatically fitted cutting-plane display patch.
    """

    def __init__(self, visibility_group_id: str) -> None:
        self._solid = OcclusionScene3D(
            visibility_group_id,
            topology_mode="closed_convex_polyhedron",
        )
        self._section_id: str | None = None
        self._plane_provider: PlaneProvider | None = None
        self._face_fill_bindings: dict[str, Mobject] = {}

    @property
    def frozen(self) -> bool:
        return self._solid.frozen

    def _require_section_mutable(self) -> None:
        if self.frozen:
            raise ConvexSectionAuthoringError(
                "visibility topology is already frozen"
            )

    def vertex(
        self,
        vertex_id: str,
        position_provider: VertexPositionProvider,
    ) -> "ConvexSectionScene3D":
        self._solid.vertex(vertex_id, position_provider)
        return self

    def face(
        self,
        face_id: str,
        vertex_ids: Sequence[str],
        *,
        occludes_strokes: bool = True,
        source_mobject: Mobject | None = None,
    ) -> "ConvexSectionScene3D":
        self._solid.face(
            face_id,
            vertex_ids,
            occludes_strokes=occludes_strokes,
        )
        if source_mobject is not None:
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
    ) -> "ConvexSectionScene3D":
        self._solid.stroke(
            source_edge_id,
            start_vertex_id,
            end_vertex_id,
            source_mobject,
            incident_face_ids=incident_face_ids,
            visibility_mode=visibility_mode,
        )
        return self

    def cutting_plane(
        self,
        section_id: str,
        plane_provider: PlaneProvider,
    ) -> "ConvexSectionScene3D":
        self._require_section_mutable()
        if self._plane_provider is not None:
            raise ConvexSectionAuthoringError(
                "convex-section v1 accepts exactly one cutting plane"
            )
        if not isinstance(section_id, str) or not section_id.strip():
            raise ConvexSectionAuthoringError(
                "section_id must be a non-empty string"
            )
        if not callable(plane_provider):
            raise ConvexSectionAuthoringError(
                "plane_provider must be callable"
            )
        try:
            sample = plane_provider()
        except Exception as exc:
            raise ConvexSectionAuthoringError(
                "plane_provider failed while freezing its initial contract"
            ) from exc
        if not isinstance(sample, SectionPlane3D):
            raise ConvexSectionAuthoringError(
                "plane_provider must return SectionPlane3D"
            )
        self._section_id = section_id.strip()
        self._plane_provider = plane_provider
        return self

    def _require_plane(self) -> tuple[str, PlaneProvider]:
        if self._section_id is None or self._plane_provider is None:
            raise ConvexSectionAuthoringError(
                "declare one cutting plane before solving or binding"
            )
        return self._section_id, self._plane_provider

    def current_positions(self) -> dict[str, Sequence[float]]:
        return self._solid.current_positions()

    @property
    def stroke_bindings(self) -> Mapping[str, Mobject]:
        return self._solid.stroke_bindings

    @property
    def face_fill_bindings(self) -> Mapping[str, Mobject]:
        return dict(self._face_fill_bindings)

    def freeze(
        self,
        *,
        tolerance_policy: TolerancePolicy | None = None,
    ) -> VisibilityModel:
        self._require_plane()
        return self._solid.freeze(tolerance_policy=tolerance_policy)

    def current_section(
        self,
        *,
        tolerance_policy: TolerancePolicy | None = None,
    ) -> ConvexSectionFrame:
        section_id, provider = self._require_plane()
        model = self.freeze(tolerance_policy=tolerance_policy)
        return intersect_plane_with_convex_polyhedron(
            section_id,
            model,
            provider(),
            vertex_positions=self.current_positions(),
            tolerance_policy=tolerance_policy,
        )

    def current_stroke_intersections(
        self,
        *,
        include_surface_edges: bool = False,
        tolerance_policy: TolerancePolicy | None = None,
    ) -> dict[str, SegmentSolidIntersection]:
        model = self.freeze(tolerance_policy=tolerance_policy)
        positions = self.current_positions()
        return {
            stroke.source_edge_id: intersect_segment_with_convex_polyhedron(
                model,
                positions[stroke.vertex_ids[0]],
                positions[stroke.vertex_ids[1]],
                vertex_positions=positions,
                tolerance_policy=tolerance_policy,
            )
            for stroke in model.strokes
            if include_surface_edges or not stroke.incident_face_ids
        }

    def controller(
        self,
        scene: object,
        *,
        projection: ParallelProjection,
        source_style: OcclusionStyle,
        section_style: ConvexSectionStyle | None = None,
        face_depth_style: FaceDepthCueStyle | None = None,
        accurate_transparency: bool = False,
        transparent_coplanar_policy: str = "section_over_solid",
        plane_patch_mode: str = "auto",
        plane_patch_margin: float = 0.15,
        display_point_provider: DisplayPointProvider | None = None,
        tolerance_policy: TolerancePolicy | None = None,
        source_coordinate_mode: str = "world",
    ) -> ConvexSection3D:
        section_id, provider = self._require_plane()
        model = self.freeze(tolerance_policy=tolerance_policy)
        expected_faces = set(model.face_map)
        bound_faces = set(self._face_fill_bindings)
        if bound_faces and bound_faces != expected_faces:
            missing = sorted(expected_faces - bound_faces)
            extra = sorted(bound_faces - expected_faces)
            raise ConvexSectionAuthoringError(
                "automatic face depth cues require a source_mobject for every face"
                + (f"; missing={missing}" if missing else "")
                + (f"; extra={extra}" if extra else "")
            )
        if face_depth_style is not None and not bound_faces:
            raise ConvexSectionAuthoringError(
                "face_depth_style requires source_mobject on every face"
            )
        if accurate_transparency and not bound_faces:
            raise ConvexSectionAuthoringError(
                "accurate_transparency requires source_mobject on every face"
            )
        return ConvexSection3D(
            scene,
            model,
            position_provider=self.current_positions,
            stroke_bindings=self.stroke_bindings,
            plane_provider=provider,
            projection=projection,
            source_style=source_style,
            section_style=section_style,
            face_fill_bindings=(
                self._face_fill_bindings if bound_faces else None
            ),
            face_depth_style=face_depth_style,
            accurate_transparency=accurate_transparency,
            transparent_coplanar_policy=transparent_coplanar_policy,
            plane_patch_mode=plane_patch_mode,
            plane_patch_margin=plane_patch_margin,
            display_point_provider=display_point_provider,
            tolerance_policy=tolerance_policy,
            source_coordinate_mode=source_coordinate_mode,
            section_id=section_id,
        )


__all__ = [
    "ConvexSectionAuthoringError",
    "ConvexSectionScene3D",
]
