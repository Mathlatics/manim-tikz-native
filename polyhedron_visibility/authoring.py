from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from manim import Mobject

from .api import AutoOcclusion3D, ParallelProjection
from .binding import DisplayPointProvider
from .contract import ContractError, TolerancePolicy, VisibilityModel
from .style import OcclusionStyle


class OcclusionAuthoringError(ValueError):
    """Raised when a manually authored visibility scene is ambiguous."""


VertexPositionProvider = Callable[[], Sequence[float]]


@dataclass(frozen=True)
class _AuthoredFace:
    face_id: str
    vertex_ids: tuple[str, ...]
    occludes_strokes: bool


@dataclass(frozen=True)
class _AuthoredStroke:
    source_edge_id: str
    vertex_ids: tuple[str, str]
    source_mobject: Mobject
    incident_face_ids: tuple[str, ...] | None
    visibility_mode: str


class OcclusionScene3D:
    """Small authoring builder for ordinary Manim scenes.

    The builder never guesses topology from a ``VGroup``.  Authors register
    stable vertex identities once, then declare maximal convex faces and the
    semantic strokes that should switch between solid and dashed rendering.
    Vertex providers remain live, so normal Manim transforms and updaters can
    move the geometry after the topology is frozen.
    """

    def __init__(
        self,
        visibility_group_id: str,
        *,
        topology_mode: str = "closed_convex_polyhedron",
    ) -> None:
        if not isinstance(visibility_group_id, str) or not visibility_group_id.strip():
            raise OcclusionAuthoringError("visibility_group_id must be a non-empty string")
        if topology_mode not in {"closed_convex_polyhedron", "independent_convex_faces"}:
            raise OcclusionAuthoringError("topology_mode is unsupported")
        self.visibility_group_id = visibility_group_id.strip()
        self.topology_mode = topology_mode
        self._vertices: dict[str, VertexPositionProvider] = {}
        self._faces: dict[str, _AuthoredFace] = {}
        self._strokes: dict[str, _AuthoredStroke] = {}
        self._model: VisibilityModel | None = None

    @property
    def frozen(self) -> bool:
        return self._model is not None

    def _require_mutable(self) -> None:
        if self.frozen:
            raise OcclusionAuthoringError("visibility topology is already frozen")

    @staticmethod
    def _identity(value: str, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise OcclusionAuthoringError(f"{label} must be a non-empty string")
        return value.strip()

    def vertex(
        self,
        vertex_id: str,
        position_provider: VertexPositionProvider,
    ) -> "OcclusionScene3D":
        self._require_mutable()
        identity = self._identity(vertex_id, "vertex_id")
        if identity in self._vertices:
            raise OcclusionAuthoringError(f"duplicate vertex_id: {identity}")
        if not callable(position_provider):
            raise OcclusionAuthoringError(f"vertex {identity} position_provider must be callable")
        self._vertices[identity] = position_provider
        return self

    def face(
        self,
        face_id: str,
        vertex_ids: Sequence[str],
        *,
        occludes_strokes: bool = True,
    ) -> "OcclusionScene3D":
        self._require_mutable()
        identity = self._identity(face_id, "face_id")
        if identity in self._faces:
            raise OcclusionAuthoringError(f"duplicate face_id: {identity}")
        vertices = tuple(self._identity(item, f"face {identity} vertex") for item in vertex_ids)
        if len(vertices) < 3 or len(set(vertices)) != len(vertices):
            raise OcclusionAuthoringError(
                f"face {identity} must contain at least three unique vertex IDs"
            )
        if not isinstance(occludes_strokes, bool):
            raise OcclusionAuthoringError(
                f"face {identity} occludes_strokes must be boolean"
            )
        self._faces[identity] = _AuthoredFace(identity, vertices, occludes_strokes)
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
    ) -> "OcclusionScene3D":
        self._require_mutable()
        identity = self._identity(source_edge_id, "source_edge_id")
        if identity in self._strokes:
            raise OcclusionAuthoringError(f"duplicate source_edge_id: {identity}")
        start = self._identity(start_vertex_id, f"stroke {identity} start")
        end = self._identity(end_vertex_id, f"stroke {identity} end")
        if start == end:
            raise OcclusionAuthoringError(f"stroke {identity} endpoints must differ")
        if not isinstance(source_mobject, Mobject):
            raise OcclusionAuthoringError(f"stroke {identity} source_mobject must be a Manim Mobject")
        incidents = None
        if incident_face_ids is not None:
            incidents = tuple(
                sorted(
                    self._identity(item, f"stroke {identity} incident face")
                    for item in incident_face_ids
                )
            )
            if len(set(incidents)) != len(incidents):
                raise OcclusionAuthoringError(
                    f"stroke {identity} incident_face_ids contains duplicates"
                )
        if visibility_mode not in {"auto", "always_visible", "always_hidden"}:
            raise OcclusionAuthoringError(f"stroke {identity} visibility_mode is unsupported")
        self._strokes[identity] = _AuthoredStroke(
            identity,
            (start, end),
            source_mobject,
            incidents,
            visibility_mode,
        )
        return self

    def current_positions(self) -> dict[str, Sequence[float]]:
        return {
            vertex_id: self._vertices[vertex_id]()
            for vertex_id in sorted(self._vertices)
        }

    def freeze(
        self,
        *,
        tolerance_policy: TolerancePolicy | None = None,
    ) -> VisibilityModel:
        if self._model is not None:
            return self._model
        face_vertex_sets = {
            face_id: set(face.vertex_ids) for face_id, face in self._faces.items()
        }
        strokes: list[dict[str, object]] = []
        for stroke in sorted(self._strokes.values(), key=lambda item: item.source_edge_id):
            incidents = stroke.incident_face_ids
            if incidents is None:
                endpoints = set(stroke.vertex_ids)
                incidents = tuple(
                    sorted(
                        face_id
                        for face_id, vertices in face_vertex_sets.items()
                        if endpoints.issubset(vertices)
                    )
                )
            strokes.append(
                {
                    "sourceEdgeId": stroke.source_edge_id,
                    "vertexIds": list(stroke.vertex_ids),
                    "incidentFaceIds": list(incidents),
                    "visibilityMode": stroke.visibility_mode,
                }
            )
        payload = {
            "schema": "manim-convex-polyhedron-visibility/v1",
            "visibilityGroupId": self.visibility_group_id,
            "vertices": [
                {
                    "vertexId": vertex_id,
                    "entryPosition": list(self._vertices[vertex_id]()),
                }
                for vertex_id in sorted(self._vertices)
            ],
            "faces": [
                {
                    "faceId": face.face_id,
                    "vertexIds": list(face.vertex_ids),
                    "occludesStrokes": face.occludes_strokes,
                }
                for face in sorted(self._faces.values(), key=lambda item: item.face_id)
            ],
            "strokes": strokes,
        }
        try:
            model = VisibilityModel.from_dict(payload)
            model.validate(
                require_closed_convex_manifold=(
                    self.topology_mode == "closed_convex_polyhedron"
                ),
                tolerance_policy=tolerance_policy,
            )
        except ContractError as exc:
            raise OcclusionAuthoringError(str(exc)) from exc
        self._model = model
        return model

    @property
    def stroke_bindings(self) -> Mapping[str, Mobject]:
        return {
            edge_id: self._strokes[edge_id].source_mobject
            for edge_id in sorted(self._strokes)
        }

    def controller(
        self,
        scene: object,
        *,
        projection: ParallelProjection,
        style: OcclusionStyle,
        display_point_provider: DisplayPointProvider | None = None,
        tolerance_policy: TolerancePolicy | None = None,
    ) -> AutoOcclusion3D:
        model = self.freeze(tolerance_policy=tolerance_policy)
        return AutoOcclusion3D(
            scene,
            model,
            position_provider=self.current_positions,
            stroke_bindings=self.stroke_bindings,
            projection=projection,
            display_point_provider=display_point_provider,
            style=style,
            tolerance_policy=tolerance_policy,
            require_closed_convex_manifold=(
                self.topology_mode == "closed_convex_polyhedron"
            ),
        )


__all__ = ["OcclusionAuthoringError", "OcclusionScene3D", "VertexPositionProvider"]
