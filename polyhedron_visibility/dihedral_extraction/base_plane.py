from __future__ import annotations

from dataclasses import dataclass
from math import acos, isfinite, pi
from typing import Mapping, Sequence

import numpy as np

from ..contract import ContractError, TolerancePolicy, VisibilityModel
from .contract import RigidTransform3D


BASE_PLANE_ROTATION_SCHEMA = "manim-base-plane-rotation/v1"


class BasePlaneRotationError(ValueError):
    """Raised when a source face cannot define one stable base-plane motion."""


def _point3(value: Sequence[float], label: str) -> np.ndarray:
    try:
        result = np.asarray(tuple(float(item) for item in value), dtype=float)
    except (TypeError, ValueError) as exc:
        raise BasePlaneRotationError(
            f"{label} must contain three numeric components"
        ) from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise BasePlaneRotationError(
            f"{label} must be a finite three-component point"
        )
    return result


def _unit(value: Sequence[float], label: str, epsilon: float) -> np.ndarray:
    result = _point3(value, label)
    length = float(np.linalg.norm(result))
    if length <= epsilon:
        raise BasePlaneRotationError(f"{label} must be non-zero")
    return result / length


def _face_normal(points: Sequence[np.ndarray], epsilon: float) -> np.ndarray:
    if len(points) < 3:
        raise BasePlaneRotationError("base face must contain at least three vertices")
    anchor = points[0]
    for index in range(1, len(points) - 1):
        first = points[index] - anchor
        second = points[index + 1] - anchor
        normal = np.cross(first, second)
        length = float(np.linalg.norm(normal))
        local_scale = max(
            float(np.linalg.norm(first)),
            float(np.linalg.norm(second)),
            epsilon,
        )
        if length > epsilon * local_scale:
            return normal / length
    raise BasePlaneRotationError("base face is degenerate")


@dataclass(frozen=True)
class BasePlaneRotation3D:
    """Rotate one authored face until its outward normal becomes the target.

    For a closed solid, ``target_outward_normal=(0, 0, -1)`` makes the selected
    face a horizontal bottom face: its outward side points down while the solid
    remains above it.  By default the registered solid's geometric center
    (the centroid of all registered vertices) stays fixed throughout the
    motion.  A caller may still provide an explicit anchor when a lesson needs
    another pivot.
    """

    face_id: str
    anchor: tuple[float, float, float]
    source_outward_normal: tuple[float, float, float]
    target_outward_normal: tuple[float, float, float]
    rotation_axis: tuple[float, float, float]
    total_angle: float
    schema: str = BASE_PLANE_ROTATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BASE_PLANE_ROTATION_SCHEMA:
            raise BasePlaneRotationError("base-plane rotation schema is unsupported")
        identity = str(self.face_id or "").strip()
        if not identity:
            raise BasePlaneRotationError("face_id must be a non-empty string")
        anchor = _point3(self.anchor, "base-plane anchor")
        source = _unit(
            self.source_outward_normal,
            "source_outward_normal",
            1.0e-12,
        )
        target = _unit(
            self.target_outward_normal,
            "target_outward_normal",
            1.0e-12,
        )
        axis = _unit(self.rotation_axis, "rotation_axis", 1.0e-12)
        try:
            angle = float(self.total_angle)
        except (TypeError, ValueError) as exc:
            raise BasePlaneRotationError("total_angle must be numeric") from exc
        if not isfinite(angle) or not 0.0 <= angle <= pi:
            raise BasePlaneRotationError(
                "total_angle must be finite and between 0 and pi"
            )
        rotation = RigidTransform3D.rotation_about_axis(axis, angle)
        mapped = np.asarray(rotation.rotation, dtype=float) @ source
        if not np.allclose(mapped, target, rtol=0.0, atol=1.0e-9):
            raise BasePlaneRotationError(
                "rotation_axis and total_angle do not map the source normal "
                "to the target normal"
            )
        object.__setattr__(self, "face_id", identity)
        object.__setattr__(self, "anchor", tuple(float(item) for item in anchor))
        object.__setattr__(
            self,
            "source_outward_normal",
            tuple(float(item) for item in source),
        )
        object.__setattr__(
            self,
            "target_outward_normal",
            tuple(float(item) for item in target),
        )
        object.__setattr__(
            self,
            "rotation_axis",
            tuple(float(item) for item in axis),
        )
        object.__setattr__(self, "total_angle", angle)

    @classmethod
    def from_model(
        cls,
        model: VisibilityModel,
        face_id: str,
        *,
        vertex_positions: Mapping[str, Sequence[float]] | None = None,
        target_outward_normal: Sequence[float] = (0.0, 0.0, -1.0),
        anchor: Sequence[float] | None = None,
        tolerance_policy: TolerancePolicy | None = None,
    ) -> "BasePlaneRotation3D":
        if not isinstance(model, VisibilityModel):
            raise BasePlaneRotationError("model must be a VisibilityModel")
        identity = str(face_id or "").strip()
        if identity not in model.face_map:
            raise BasePlaneRotationError(f"unknown base face: {identity or '<empty>'}")
        policy = tolerance_policy or TolerancePolicy()
        raw_positions = model.entry_positions if vertex_positions is None else vertex_positions
        try:
            model.validate(
                vertex_positions=raw_positions,
                require_closed_convex_manifold=True,
                tolerance_policy=policy,
            )
        except ContractError as exc:
            raise BasePlaneRotationError(
                f"base-plane rotation requires a valid closed convex solid: {exc}"
            ) from exc
        face = model.face_map[identity]
        try:
            points = [
                _point3(raw_positions[vertex_id], f"vertex {vertex_id}")
                for vertex_id in face.vertex_ids
            ]
        except KeyError as exc:
            raise BasePlaneRotationError(
                f"base face {identity} is missing vertex {exc.args[0]}"
            ) from exc
        resolved = policy.resolve(points)
        source_normal = _face_normal(points, resolved.world)
        target_normal = _unit(
            target_outward_normal,
            "target_outward_normal",
            max(resolved.angular, float(np.finfo(float).eps)),
        )
        cosine = float(np.clip(np.dot(source_normal, target_normal), -1.0, 1.0))
        cross = np.cross(source_normal, target_normal)
        cross_length = float(np.linalg.norm(cross))
        angle = float(acos(cosine))
        if cross_length > resolved.angular:
            axis = cross / cross_length
        else:
            # Parallel normals need no rotation.  Anti-parallel normals have
            # infinitely many shortest half-turn axes; select the first stable
            # authored boundary direction, which lies in the source plane.
            edge = points[1] - points[0]
            axis = _unit(edge, "base face heading edge", resolved.world)
            angle = 0.0 if cosine >= 0.0 else pi
        if anchor is None:
            solid_points = np.asarray(
                [
                    _point3(raw_positions[vertex_id], f"vertex {vertex_id}")
                    for vertex_id in sorted(model.vertex_map)
                ],
                dtype=float,
            )
            pivot = np.mean(solid_points, axis=0)
        else:
            pivot = _point3(anchor, "base-plane anchor")
        return cls(
            identity,
            tuple(float(item) for item in pivot),
            tuple(float(item) for item in source_normal),
            tuple(float(item) for item in target_normal),
            tuple(float(item) for item in axis),
            angle,
        )

    def transform(self, progress: float) -> RigidTransform3D:
        try:
            value = float(progress)
        except (TypeError, ValueError) as exc:
            raise BasePlaneRotationError("base-plane progress must be numeric") from exc
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise BasePlaneRotationError(
                "base-plane progress must be finite and between 0 and 1"
            )
        if self.total_angle == 0.0:
            return RigidTransform3D.identity()
        return RigidTransform3D.rotation_about_axis(
            self.rotation_axis,
            self.total_angle * value,
            about_point=self.anchor,
        )

    def final_transform(self) -> RigidTransform3D:
        return self.transform(1.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "faceId": self.face_id,
            "anchor": list(self.anchor),
            "sourceOutwardNormal": list(self.source_outward_normal),
            "targetOutwardNormal": list(self.target_outward_normal),
            "rotationAxis": list(self.rotation_axis),
            "totalAngle": self.total_angle,
        }


__all__ = [
    "BASE_PLANE_ROTATION_SCHEMA",
    "BasePlaneRotation3D",
    "BasePlaneRotationError",
]
