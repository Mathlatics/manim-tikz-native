from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

from manim import Mobject

from ..api import ParallelProjection
from ..binding import DisplayPointProvider
from ..contract import TolerancePolicy
from ..style import OcclusionStyle
from .contract import (
    ARTICULATED_HINGE_POLICY,
    OPEN_FACE_MODEL_SCHEMA,
    OPEN_FACE_TOPOLOGY,
    OpenFaceContractError,
    OpenFaceVisibilityModel,
)
from .manim import OpenFaceOcclusion3D


class OpenFaceAuthoringError(ValueError):
    """Raised when an ordinary Manim open-face scene is ambiguous."""


OpenFaceVertexPositionProvider = Callable[[], Sequence[float]]


@dataclass(frozen=True)
class _AuthoredOpenFace:
    face_id: str
    logical_surface_id: str
    vertex_ids: tuple[str, ...]
    occludes_strokes: bool
    source_mobject: Mobject | None


@dataclass(frozen=True)
class _AuthoredOpenFaceSeam:
    seam_id: str
    face_ids: tuple[str, str]
    vertex_ids: tuple[str, str]


@dataclass(frozen=True)
class _AuthoredOpenFaceStroke:
    source_edge_id: str
    vertex_ids: tuple[str, str]
    source_mobject: Mobject
    incident_face_ids: tuple[str, ...] | None
    excluded_occluder_face_ids: tuple[str, ...]
    visibility_mode: str


class OpenFaceScene3D:
    """Author finite convex panels and articulated hinges in ordinary Manim.

    Geometry providers stay live after ``freeze()``.  The topology is fixed,
    while normal Manim trackers, transforms, and updaters may continue moving
    the registered vertices.
    """

    def __init__(self, visibility_group_id: str) -> None:
        if not isinstance(visibility_group_id, str) or not visibility_group_id.strip():
            raise OpenFaceAuthoringError(
                "visibility_group_id must be a non-empty string"
            )
        self.visibility_group_id = visibility_group_id.strip()
        self._vertices: dict[str, OpenFaceVertexPositionProvider] = {}
        self._faces: dict[str, _AuthoredOpenFace] = {}
        self._seams: dict[str, _AuthoredOpenFaceSeam] = {}
        self._strokes: dict[str, _AuthoredOpenFaceStroke] = {}
        self._model: OpenFaceVisibilityModel | None = None

    @property
    def frozen(self) -> bool:
        return self._model is not None

    def _require_mutable(self) -> None:
        if self.frozen:
            raise OpenFaceAuthoringError("open-face visibility topology is already frozen")

    @staticmethod
    def _identity(value: str, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise OpenFaceAuthoringError(f"{label} must be a non-empty string")
        return value.strip()

    @classmethod
    def _identity_tuple(
        cls,
        values: Sequence[str],
        label: str,
        *,
        minimum: int = 0,
        exact: int | None = None,
    ) -> tuple[str, ...]:
        result = tuple(cls._identity(item, label) for item in values)
        if len(result) < minimum:
            raise OpenFaceAuthoringError(
                f"{label} must contain at least {minimum} identities"
            )
        if exact is not None and len(result) != exact:
            raise OpenFaceAuthoringError(
                f"{label} must contain exactly {exact} identities"
            )
        if len(set(result)) != len(result):
            raise OpenFaceAuthoringError(f"{label} contains duplicate identities")
        return result

    def vertex(
        self,
        vertex_id: str,
        position_provider: OpenFaceVertexPositionProvider,
    ) -> "OpenFaceScene3D":
        self._require_mutable()
        identity = self._identity(vertex_id, "vertex_id")
        if identity in self._vertices:
            raise OpenFaceAuthoringError(f"duplicate vertex_id: {identity}")
        if not callable(position_provider):
            raise OpenFaceAuthoringError(
                f"vertex {identity} position_provider must be callable"
            )
        self._vertices[identity] = position_provider
        return self

    def face(
        self,
        face_id: str,
        vertex_ids: Sequence[str],
        *,
        logical_surface_id: str,
        occludes_strokes: bool = True,
        source_mobject: Mobject | None = None,
    ) -> "OpenFaceScene3D":
        self._require_mutable()
        identity = self._identity(face_id, "face_id")
        if identity in self._faces:
            raise OpenFaceAuthoringError(f"duplicate face_id: {identity}")
        surface = self._identity(logical_surface_id, f"face {identity} logical_surface_id")
        vertices = self._identity_tuple(
            vertex_ids,
            f"face {identity} vertex_ids",
            minimum=3,
        )
        if not isinstance(occludes_strokes, bool):
            raise OpenFaceAuthoringError(
                f"face {identity} occludes_strokes must be boolean"
            )
        if source_mobject is not None and not isinstance(source_mobject, Mobject):
            raise OpenFaceAuthoringError(
                f"face {identity} source_mobject must be a Manim Mobject"
            )
        self._faces[identity] = _AuthoredOpenFace(
            identity,
            surface,
            vertices,
            occludes_strokes,
            source_mobject,
        )
        return self

    def articulated_hinge(
        self,
        seam_id: str,
        first_face_id: str,
        second_face_id: str,
        start_vertex_id: str,
        end_vertex_id: str,
    ) -> "OpenFaceScene3D":
        self._require_mutable()
        identity = self._identity(seam_id, "seam_id")
        if identity in self._seams:
            raise OpenFaceAuthoringError(f"duplicate seam_id: {identity}")
        faces = self._identity_tuple(
            (first_face_id, second_face_id),
            f"seam {identity} face_ids",
            exact=2,
        )
        vertices = self._identity_tuple(
            (start_vertex_id, end_vertex_id),
            f"seam {identity} vertex_ids",
            exact=2,
        )
        self._seams[identity] = _AuthoredOpenFaceSeam(
            identity,
            tuple(sorted((faces[0], faces[1]))),
            (vertices[0], vertices[1]),
        )
        return self

    def hinge(
        self,
        seam_id: str,
        first_face_id: str,
        second_face_id: str,
        start_vertex_id: str,
        end_vertex_id: str,
    ) -> "OpenFaceScene3D":
        """Short authoring alias for an ``articulated_hinge`` seam."""

        return self.articulated_hinge(
            seam_id,
            first_face_id,
            second_face_id,
            start_vertex_id,
            end_vertex_id,
        )

    def stroke(
        self,
        source_edge_id: str,
        start_vertex_id: str,
        end_vertex_id: str,
        source_mobject: Mobject,
        *,
        incident_face_ids: Sequence[str] | None = None,
        excluded_occluder_face_ids: Sequence[str] = (),
        visibility_mode: str = "auto",
    ) -> "OpenFaceScene3D":
        self._require_mutable()
        identity = self._identity(source_edge_id, "source_edge_id")
        if identity in self._strokes:
            raise OpenFaceAuthoringError(f"duplicate source_edge_id: {identity}")
        vertices = self._identity_tuple(
            (start_vertex_id, end_vertex_id),
            f"stroke {identity} endpoints",
            exact=2,
        )
        if not isinstance(source_mobject, Mobject):
            raise OpenFaceAuthoringError(
                f"stroke {identity} source_mobject must be a Manim Mobject"
            )
        incidents = None
        if incident_face_ids is not None:
            incidents = tuple(
                sorted(
                    self._identity_tuple(
                        incident_face_ids,
                        f"stroke {identity} incident_face_ids",
                    )
                )
            )
        excluded = tuple(
            sorted(
                self._identity_tuple(
                    excluded_occluder_face_ids,
                    f"stroke {identity} excluded_occluder_face_ids",
                )
            )
        )
        if set(incidents or ()) & set(excluded):
            raise OpenFaceAuthoringError(
                f"stroke {identity} cannot mark one face both incident and excluded"
            )
        if visibility_mode not in {"auto", "always_visible", "always_hidden"}:
            raise OpenFaceAuthoringError(
                f"stroke {identity} visibility_mode is unsupported"
            )
        self._strokes[identity] = _AuthoredOpenFaceStroke(
            identity,
            (vertices[0], vertices[1]),
            source_mobject,
            incidents,
            excluded,
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
    ) -> OpenFaceVisibilityModel:
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
                    "excludedOccluderFaceIds": list(
                        stroke.excluded_occluder_face_ids
                    ),
                    "visibilityMode": stroke.visibility_mode,
                }
            )
        payload = {
            "schema": OPEN_FACE_MODEL_SCHEMA,
            "topology": OPEN_FACE_TOPOLOGY,
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
                    "logicalSurfaceId": face.logical_surface_id,
                    "vertexIds": list(face.vertex_ids),
                    "occludesStrokes": face.occludes_strokes,
                }
                for face in sorted(self._faces.values(), key=lambda item: item.face_id)
            ],
            "seams": [
                {
                    "seamId": seam.seam_id,
                    "policy": ARTICULATED_HINGE_POLICY,
                    "faceIds": list(seam.face_ids),
                    "vertexIds": list(seam.vertex_ids),
                }
                for seam in sorted(self._seams.values(), key=lambda item: item.seam_id)
            ],
            "strokes": strokes,
        }
        try:
            model = OpenFaceVisibilityModel.from_dict(payload)
            model.validate(
                vertex_positions=self.current_positions(),
                tolerance_policy=tolerance_policy,
            )
        except OpenFaceContractError as exc:
            raise OpenFaceAuthoringError(str(exc)) from exc
        self._model = model
        return model

    @property
    def stroke_bindings(self) -> Mapping[str, Mobject]:
        return {
            edge_id: self._strokes[edge_id].source_mobject
            for edge_id in sorted(self._strokes)
        }

    @property
    def face_fill_bindings(self) -> Mapping[str, Mobject] | None:
        values = {
            face_id: face.source_mobject
            for face_id, face in self._faces.items()
            if face.source_mobject is not None
        }
        if not values:
            return None
        if len(values) != len(self._faces):
            raise OpenFaceAuthoringError(
                "automatic face fill ordering requires a source_mobject for every face"
            )
        return {face_id: values[face_id] for face_id in sorted(values)}  # type: ignore[return-value]

    def controller(
        self,
        scene: object,
        *,
        projection: ParallelProjection,
        style: OcclusionStyle,
        display_point_provider: DisplayPointProvider | None = None,
        tolerance_policy: TolerancePolicy | None = None,
        source_coordinate_mode: Literal["world", "display"] = "world",
    ) -> OpenFaceOcclusion3D:
        model = self.freeze(tolerance_policy=tolerance_policy)
        return OpenFaceOcclusion3D(
            scene,
            model,
            position_provider=self.current_positions,
            stroke_bindings=self.stroke_bindings,
            face_fill_bindings=self.face_fill_bindings,
            projection=projection,
            display_point_provider=display_point_provider,
            style=style,
            tolerance_policy=tolerance_policy,
            source_coordinate_mode=source_coordinate_mode,
        )


__all__ = [
    "OpenFaceAuthoringError",
    "OpenFaceScene3D",
    "OpenFaceVertexPositionProvider",
]
