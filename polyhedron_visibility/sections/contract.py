from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np


SECTION_PLANE_SCHEMA = "manim-convex-polyhedron-section-plane/v1"


class ConvexSectionContractError(ValueError):
    """Raised when a cutting-plane contract is ambiguous or non-finite."""


def _point3(value: object, label: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise ConvexSectionContractError(f"{label} must contain three numbers")
    result = tuple(float(item) for item in value)
    if not all(isfinite(item) for item in result):
        raise ConvexSectionContractError(f"{label} must be finite")
    return result


def _positive(value: object, label: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise ConvexSectionContractError(f"{label} must be finite and positive")
    return result


def _strict_keys(payload: Mapping[str, object], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ConvexSectionContractError(
            f"{label} contains unsupported fields: {', '.join(unknown)}"
        )


@dataclass(frozen=True)
class SectionPlane3D:
    """One plane equation plus an authored rectangular display patch.

    ``normal`` and ``u_axis`` are canonicalized to an orthonormal frame.  A
    caller animating the plane should provide an explicit ``u_axis`` when it
    needs the rectangular patch to keep one authored in-plane orientation.
    The realtime section controller treats the half-extents as minimum display
    dimensions by default and auto-fits them around the complete solid; its
    strict mode preserves the literal finite rectangle.
    """

    plane_id: str
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    half_width: float
    half_height: float
    u_axis: tuple[float, float, float] | None = None
    occludes_strokes: bool = True
    schema: str = SECTION_PLANE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SECTION_PLANE_SCHEMA:
            raise ConvexSectionContractError(
                f"section plane schema must be {SECTION_PLANE_SCHEMA}"
            )
        if not isinstance(self.plane_id, str) or not self.plane_id.strip():
            raise ConvexSectionContractError("plane_id must be a non-empty string")
        point = np.asarray(_point3(self.point, "plane point"), dtype=float)
        normal = np.asarray(_point3(self.normal, "plane normal"), dtype=float)
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= 1.0e-14:
            raise ConvexSectionContractError("plane normal must be non-zero")
        normal = normal / normal_length

        if self.u_axis is None:
            basis = np.eye(3)[int(np.argmin(np.abs(normal)))]
            u_axis = np.cross(normal, basis)
        else:
            authored = np.asarray(_point3(self.u_axis, "plane u_axis"), dtype=float)
            u_axis = authored - float(np.dot(authored, normal)) * normal
        u_length = float(np.linalg.norm(u_axis))
        if u_length <= 1.0e-14:
            raise ConvexSectionContractError(
                "plane u_axis must not be parallel to the normal"
            )
        u_axis = u_axis / u_length

        if not isinstance(self.occludes_strokes, bool):
            raise ConvexSectionContractError("occludes_strokes must be boolean")
        object.__setattr__(self, "point", tuple(float(item) for item in point))
        object.__setattr__(self, "normal", tuple(float(item) for item in normal))
        object.__setattr__(self, "u_axis", tuple(float(item) for item in u_axis))
        object.__setattr__(
            self, "half_width", _positive(self.half_width, "plane half_width")
        )
        object.__setattr__(
            self, "half_height", _positive(self.half_height, "plane half_height")
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SectionPlane3D":
        if not isinstance(payload, Mapping):
            raise ConvexSectionContractError("section plane must be an object")
        _strict_keys(
            payload,
            {
                "schema",
                "planeId",
                "point",
                "normal",
                "uAxis",
                "halfWidth",
                "halfHeight",
                "occludesStrokes",
            },
            "section plane",
        )
        if payload.get("schema") != SECTION_PLANE_SCHEMA:
            raise ConvexSectionContractError(
                f"section plane schema must be {SECTION_PLANE_SCHEMA}"
            )
        return cls(
            plane_id=str(payload.get("planeId", "")),
            point=_point3(payload.get("point"), "plane point"),
            normal=_point3(payload.get("normal"), "plane normal"),
            u_axis=(
                None
                if payload.get("uAxis") is None
                else _point3(payload.get("uAxis"), "plane uAxis")
            ),
            half_width=_positive(payload.get("halfWidth"), "plane halfWidth"),
            half_height=_positive(payload.get("halfHeight"), "plane halfHeight"),
            occludes_strokes=payload.get("occludesStrokes", True),
        )

    @property
    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normal = np.asarray(self.normal, dtype=float)
        u_axis = np.asarray(self.u_axis, dtype=float)
        v_axis = np.cross(normal, u_axis)
        v_axis = v_axis / float(np.linalg.norm(v_axis))
        return u_axis, v_axis, normal

    def signed_distance(self, value: Sequence[float]) -> float:
        point = np.asarray(_point3(value, "plane query point"), dtype=float)
        return float(np.dot(point - np.asarray(self.point, dtype=float), self.normal))

    def coordinates_in_plane(self, value: Sequence[float]) -> tuple[float, float]:
        point = np.asarray(_point3(value, "plane query point"), dtype=float)
        delta = point - np.asarray(self.point, dtype=float)
        u_axis, v_axis, _normal = self.basis
        return float(np.dot(delta, u_axis)), float(np.dot(delta, v_axis))

    def patch_corners(self) -> tuple[tuple[float, float, float], ...]:
        center = np.asarray(self.point, dtype=float)
        u_axis, v_axis, _normal = self.basis
        corners = (
            center - self.half_width * u_axis - self.half_height * v_axis,
            center + self.half_width * u_axis - self.half_height * v_axis,
            center + self.half_width * u_axis + self.half_height * v_axis,
            center - self.half_width * u_axis + self.half_height * v_axis,
        )
        return tuple(tuple(float(item) for item in point) for point in corners)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "planeId": self.plane_id,
            "point": list(self.point),
            "normal": list(self.normal),
            "uAxis": list(self.u_axis or ()),
            "halfWidth": self.half_width,
            "halfHeight": self.half_height,
            "occludesStrokes": self.occludes_strokes,
        }


__all__ = [
    "ConvexSectionContractError",
    "SECTION_PLANE_SCHEMA",
    "SectionPlane3D",
]
