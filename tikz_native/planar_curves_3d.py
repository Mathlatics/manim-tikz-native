"""Static TikZ semantics for explicitly supported planar curves in 3D.

Ordinary TikZ ``circle`` and ``ellipse`` paths are two-dimensional: a center
and radii do not determine a supporting plane in world space.  The restricted
3D frontend therefore authors a plane from three named coordinates ``O/U/V``:

* ``O`` is the plane origin;
* ``O -> U`` fixes the positive parameter-phase direction;
* ``V`` fixes the positive side of the in-plane ``v`` axis and consequently
  the orientation of the plane normal.

This module is deliberately renderer-neutral and does not import the compiler.
The compiler can translate :class:`PlanarTikz3DError` into its public error
type without creating an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, tau
from typing import Mapping, Sequence

import numpy as np

from polyhedron_visibility.quadrics.planar_curves import (
    Circle3DSpec,
    Ellipse3DSpec,
    PlanarCurve3DContractError,
    PlanarCurve3DSpec,
    PlanarFrame3D,
)


_PLANE_POINT_COUNT = 3
_DIRECTION_RELATIVE_TOLERANCE = float(np.sqrt(np.finfo(float).eps))
_FRAME_GEOMETRY_FIELDS = frozenset(
    {"plane_id", "plane_point_names", "frame", "static"}
)
_GEOMETRY_FIELDS = frozenset(
    {"plane_id", "plane_point_names", "frame", "curve", "static"}
)


class PlanarTikz3DError(ValueError):
    """Raised when explicit TikZ 3D planar semantics cannot be certified."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanarTikz3DError(f"{label} must be a non-empty string")
    return value.strip()


def _plane_point_names(
    value: object,
    *,
    canonical_payload: bool = False,
) -> tuple[str, str, str]:
    if canonical_payload:
        if not isinstance(value, list):
            raise PlanarTikz3DError(
                "canonical plane_point_names must be a JSON array"
            )
    elif isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PlanarTikz3DError(
            "plane_point_names must contain the three names O/U/V"
        )
    if not isinstance(value, Sequence) or len(value) != _PLANE_POINT_COUNT:
        raise PlanarTikz3DError(
            "plane_point_names must contain exactly three coordinate names O/U/V"
        )
    names = tuple(
        _identity(item, f"plane point name {index}")
        for index, item in enumerate(value)
    )
    if len(set(names)) != _PLANE_POINT_COUNT:
        raise PlanarTikz3DError(
            "plane_point_names must identify three distinct coordinates"
        )
    return names  # type: ignore[return-value]


def _point3(value: object, label: str) -> np.ndarray:
    if isinstance(value, (str, bytes)):
        raise PlanarTikz3DError(f"{label} must contain three finite numbers")
    try:
        authored = np.asarray(value, dtype=object)
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PlanarTikz3DError(
            f"{label} must contain three finite numbers"
        ) from exc
    if (
        authored.shape != (3,)
        or result.shape != (3,)
        or any(isinstance(item, (bool, np.bool_)) for item in authored)
        or not np.all(np.isfinite(result))
    ):
        raise PlanarTikz3DError(f"{label} must contain three finite numbers")
    return result


def _named_point(
    coordinates: Mapping[str, Sequence[float]],
    name: str,
) -> np.ndarray:
    if name not in coordinates:
        raise PlanarTikz3DError(
            f"named 3D plane references unknown coordinate {name!r}"
        )
    return _point3(coordinates[name], f"coordinate {name!r}")


def _scaled_direction(value: np.ndarray, label: str) -> np.ndarray:
    if not np.all(np.isfinite(value)):
        raise PlanarTikz3DError(
            f"{label} must remain finite after subtracting O"
        )
    scale = float(np.max(np.abs(value)))
    if not isfinite(scale) or scale <= 0.0:
        raise PlanarTikz3DError(f"{label} must have non-zero length")
    result = value / scale
    if not np.all(np.isfinite(result)):
        raise PlanarTikz3DError(f"{label} lies outside the finite numeric range")
    return result


def frame_from_named_points(
    frame_id: str,
    plane_point_names: Sequence[str],
    coordinates: Mapping[str, Sequence[float]],
) -> PlanarFrame3D:
    """Certify a right-handed plane frame from named coordinates ``O/U/V``.

    The operation is scale-aware: the two authored directions are normalized
    before taking their cross product.  Nearly collinear triples fail explicitly
    instead of manufacturing an unstable normal.
    """

    identity = _identity(frame_id, "plane_id")
    names = _plane_point_names(plane_point_names)
    if not isinstance(coordinates, Mapping):
        raise PlanarTikz3DError("named 3D coordinates must be a mapping")
    origin = _named_point(coordinates, names[0])
    u_point = _named_point(coordinates, names[1])
    v_point = _named_point(coordinates, names[2])
    with np.errstate(all="ignore"):
        u_delta = u_point - origin
        v_delta = v_point - origin
    u_direction = _scaled_direction(u_delta, "O->U direction")
    v_reference = _scaled_direction(v_delta, "O->V direction")
    with np.errstate(all="ignore"):
        normal_seed = np.cross(u_direction, v_reference)
        cross_length = float(np.linalg.norm(normal_seed))
        direction_product = float(
            np.linalg.norm(u_direction) * np.linalg.norm(v_reference)
        )
    if (
        not np.all(np.isfinite(normal_seed))
        or not isfinite(cross_length)
        or not isfinite(direction_product)
        or cross_length <= _DIRECTION_RELATIVE_TOLERANCE * direction_product
    ):
        raise PlanarTikz3DError(
            "named 3D plane points O/U/V must be non-collinear and numerically distinguishable"
        )
    try:
        frame = PlanarFrame3D(
            identity,
            tuple(float(item) for item in origin),
            tuple(float(item) for item in normal_seed),
            u_axis=tuple(float(item) for item in u_direction),
        )
    except PlanarCurve3DContractError as exc:
        raise PlanarTikz3DError(
            f"named 3D plane {identity!r} cannot be certified: {exc}"
        ) from exc

    # The cross-product order O->U x O->V must place V in the positive
    # in-plane v half-space.  Keep this as explicit evidence rather than
    # relying on an implementation detail of AffineFrame3D.
    positive_v = float(np.dot(np.asarray(frame.v_axis), v_reference))
    v_tolerance = (
        _DIRECTION_RELATIVE_TOLERANCE
        * float(np.linalg.norm(v_reference))
    )
    if not isfinite(positive_v) or positive_v <= v_tolerance:
        raise PlanarTikz3DError(
            "named 3D plane orientation cannot certify V on the positive v side"
        )
    return frame


def circle_from_plane_coordinates(
    curve_id: str,
    frame: PlanarFrame3D,
    center_coordinates: Sequence[float],
    radius: float,
) -> Circle3DSpec:
    """Author one full circle from coordinates in a certified named plane."""

    if not isinstance(frame, PlanarFrame3D):
        raise TypeError("frame must be a PlanarFrame3D")
    try:
        return Circle3DSpec.from_plane_coordinates(
            curve_id,
            frame,
            center_coordinates,
            radius,
        )
    except PlanarCurve3DContractError as exc:
        raise PlanarTikz3DError(
            f"TikZ 3D circle {curve_id!r} cannot be certified: {exc}"
        ) from exc


def ellipse_from_plane_coordinates(
    curve_id: str,
    frame: PlanarFrame3D,
    center_coordinates: Sequence[float],
    semi_u: float,
    semi_v: float,
) -> Ellipse3DSpec:
    """Author one full ellipse aligned with a certified named plane frame."""

    if not isinstance(frame, PlanarFrame3D):
        raise TypeError("frame must be a PlanarFrame3D")
    try:
        return Ellipse3DSpec.from_plane_coordinates(
            curve_id,
            frame,
            center_coordinates,
            semi_u,
            semi_v,
        )
    except PlanarCurve3DContractError as exc:
        raise PlanarTikz3DError(
            f"TikZ 3D ellipse {curve_id!r} cannot be certified: {exc}"
        ) from exc


@dataclass(frozen=True, slots=True)
class PlanarFrameGeometry3D:
    """Strict compiler payload for one named static supporting plane."""

    plane_id: str
    plane_point_names: tuple[str, str, str]
    frame: PlanarFrame3D
    static: bool = True

    def __post_init__(self) -> None:
        plane_id = _identity(self.plane_id, "plane_id")
        names = _plane_point_names(self.plane_point_names)
        if not isinstance(self.frame, PlanarFrame3D):
            raise TypeError("frame must be a PlanarFrame3D")
        if self.static is not True:
            raise PlanarTikz3DError(
                "planar TikZ 3D frame must explicitly declare static=true"
            )
        if plane_id != self.frame.frame_id:
            raise PlanarTikz3DError(
                "plane_id disagrees with the restored supporting frame identity"
            )
        object.__setattr__(self, "plane_id", plane_id)
        object.__setattr__(self, "plane_point_names", names)

    def to_dict(self) -> dict[str, object]:
        return {
            "plane_id": self.plane_id,
            "plane_point_names": list(self.plane_point_names),
            "frame": self.frame.to_dict(),
            "static": True,
        }


def planar_frame_geometry_payload(
    frame: PlanarFrame3D,
    plane_point_names: Sequence[str],
) -> dict[str, object]:
    """Return canonical compiler evidence for a named supporting plane."""

    return PlanarFrameGeometry3D(
        frame.frame_id,
        _plane_point_names(plane_point_names),
        frame,
    ).to_dict()


def restore_planar_frame_geometry(
    payload: Mapping[str, object],
    *,
    expected_plane_id: str | None = None,
) -> PlanarFrameGeometry3D:
    """Strictly restore one named supporting-plane declaration."""

    if not isinstance(payload, Mapping):
        raise PlanarTikz3DError("planar TikZ 3D frame geometry must be an object")
    keys = frozenset(payload.keys())
    if keys != _FRAME_GEOMETRY_FIELDS:
        missing = sorted(key for key in _FRAME_GEOMETRY_FIELDS if key not in keys)
        extra = sorted(
            repr(key) for key in keys if key not in _FRAME_GEOMETRY_FIELDS
        )
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise PlanarTikz3DError(
            "planar TikZ 3D frame geometry fields are invalid: "
            + "; ".join(details)
        )
    if payload["static"] is not True:
        raise PlanarTikz3DError(
            "planar TikZ 3D frame must explicitly declare static=true"
        )
    plane_id = _identity(payload["plane_id"], "plane_id")
    names = _plane_point_names(
        payload["plane_point_names"],
        canonical_payload=True,
    )
    raw_frame = payload["frame"]
    if not isinstance(raw_frame, Mapping):
        raise PlanarTikz3DError("canonical planar frame must be an object")
    try:
        frame = PlanarFrame3D.from_dict(raw_frame)
    except (PlanarCurve3DContractError, TypeError) as exc:
        raise PlanarTikz3DError(
            f"canonical planar frame geometry cannot be restored: {exc}"
        ) from exc
    result = PlanarFrameGeometry3D(plane_id, names, frame)
    if result.to_dict() != dict(payload):
        raise PlanarTikz3DError(
            "planar frame payload must already be in canonical form"
        )
    if expected_plane_id is not None and result.plane_id != _identity(
        expected_plane_id,
        "expected_plane_id",
    ):
        raise PlanarTikz3DError(
            "restored planar frame identity disagrees with its registry identity"
        )
    return result


@dataclass(frozen=True, slots=True)
class PlanarCurveGeometry3D:
    """Strict restored form of one static compiler geometry payload."""

    plane_id: str
    plane_point_names: tuple[str, str, str]
    frame: PlanarFrame3D
    curve: PlanarCurve3DSpec
    static: bool = True

    def __post_init__(self) -> None:
        plane_id = _identity(self.plane_id, "plane_id")
        names = _plane_point_names(self.plane_point_names)
        if not isinstance(self.frame, PlanarFrame3D):
            raise TypeError("frame must be a PlanarFrame3D")
        if not isinstance(self.curve, (Circle3DSpec, Ellipse3DSpec)):
            raise TypeError("curve must be a Circle3DSpec or Ellipse3DSpec")
        if self.static is not True:
            raise PlanarTikz3DError(
                "planar TikZ 3D geometry must explicitly declare static=true"
            )
        if plane_id != self.frame.frame_id:
            raise PlanarTikz3DError(
                "plane_id disagrees with the restored supporting frame identity"
            )
        if self.curve.frame != self.frame:
            raise PlanarTikz3DError(
                "planar curve disagrees with its restored supporting frame"
            )
        if self.curve.curve_id == plane_id:
            raise PlanarTikz3DError(
                "plane and curve identities must be distinct"
            )
        if (
            self.curve.domain.start != 0.0
            or self.curve.domain.end != tau
        ):
            raise PlanarTikz3DError(
                "explicit TikZ 3D planar curve v1 requires one complete revolution"
            )
        object.__setattr__(self, "plane_id", plane_id)
        object.__setattr__(self, "plane_point_names", names)

    def to_dict(self) -> dict[str, object]:
        return planar_curve_geometry_payload(
            self.frame,
            self.curve,
            self.plane_point_names,
        )


def planar_curve_geometry_payload(
    frame: PlanarFrame3D,
    curve: PlanarCurve3DSpec,
    plane_point_names: Sequence[str],
) -> dict[str, object]:
    """Return the canonical compiler geometry for one static planar curve."""

    if not isinstance(frame, PlanarFrame3D):
        raise TypeError("frame must be a PlanarFrame3D")
    if not isinstance(curve, (Circle3DSpec, Ellipse3DSpec)):
        raise TypeError("curve must be a Circle3DSpec or Ellipse3DSpec")
    restored = PlanarCurveGeometry3D(
        frame.frame_id,
        _plane_point_names(plane_point_names),
        frame,
        curve,
    )
    return {
        "plane_id": restored.plane_id,
        "plane_point_names": list(restored.plane_point_names),
        "frame": restored.frame.to_dict(),
        "curve": restored.curve.to_dict(),
        "static": True,
    }


def restore_planar_curve_geometry(
    payload: Mapping[str, object],
    *,
    expected_curve_id: str | None = None,
) -> PlanarCurveGeometry3D:
    """Strictly restore frame and curve from canonical compiler geometry.

    Missing or additional fields are rejected.  The nested core payloads are
    delegated to their own strict ``from_dict`` implementations, including the
    persistent direction-seed checks on :class:`PlanarFrame3D`.
    """

    if not isinstance(payload, Mapping):
        raise PlanarTikz3DError("planar TikZ 3D geometry must be an object")
    keys = frozenset(payload.keys())
    if keys != _GEOMETRY_FIELDS:
        missing = sorted(key for key in _GEOMETRY_FIELDS if key not in keys)
        extra = sorted(repr(key) for key in keys if key not in _GEOMETRY_FIELDS)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise PlanarTikz3DError(
            "planar TikZ 3D geometry fields are invalid: " + "; ".join(details)
        )
    if payload["static"] is not True:
        raise PlanarTikz3DError(
            "planar TikZ 3D geometry must explicitly declare static=true"
        )
    frame_geometry = restore_planar_frame_geometry(
        {
            "plane_id": payload["plane_id"],
            "plane_point_names": payload["plane_point_names"],
            "frame": payload["frame"],
            "static": payload["static"],
        }
    )
    plane_id = frame_geometry.plane_id
    names = frame_geometry.plane_point_names
    raw_frame = payload["frame"]
    raw_curve = payload["curve"]
    if not isinstance(raw_frame, Mapping):
        raise PlanarTikz3DError("canonical planar frame must be an object")
    if not isinstance(raw_curve, Mapping):
        raise PlanarTikz3DError("canonical planar curve must be an object")
    try:
        frame = frame_geometry.frame
        kind = raw_curve.get("kind")
        if kind == "circle":
            curve: PlanarCurve3DSpec = Circle3DSpec.from_dict(raw_curve, frame)
        elif kind == "ellipse":
            curve = Ellipse3DSpec.from_dict(raw_curve, frame)
        else:
            raise PlanarTikz3DError(
                "canonical planar curve kind must be 'circle' or 'ellipse'"
            )
    except PlanarCurve3DContractError as exc:
        raise PlanarTikz3DError(
            f"canonical planar curve geometry cannot be restored: {exc}"
        ) from exc
    except TypeError as exc:
        raise PlanarTikz3DError(
            f"canonical planar curve geometry has invalid field types: {exc}"
        ) from exc
    try:
        payload_is_canonical = (
            frame.to_dict() == dict(raw_frame)
            and curve.to_dict() == dict(raw_curve)
        )
    except (TypeError, ValueError):
        payload_is_canonical = False
    if not payload_is_canonical:
        raise PlanarTikz3DError(
            "planar frame and curve payloads must already be in canonical form"
        )
    result = PlanarCurveGeometry3D(plane_id, names, frame, curve)
    if expected_curve_id is not None:
        identity = _identity(expected_curve_id, "expected_curve_id")
        if result.curve.curve_id != identity:
            raise PlanarTikz3DError(
                "restored planar curve identity disagrees with its compiler object identity"
            )
    return result


def restore_registered_planar_curve_geometry(
    payload: Mapping[str, object],
    frame_registry: Mapping[str, object],
    *,
    expected_curve_id: str | None = None,
) -> PlanarCurveGeometry3D:
    """Restore a curve and certify it against its picture-level plane entry.

    The object payload intentionally repeats the immutable frame so a saved
    object is self-describing.  That duplication must not become two competing
    authorities: renderers and motion analysis call this function to prove the
    object bytes still agree with the named plane registry in its picture.
    """

    if not isinstance(frame_registry, Mapping):
        raise PlanarTikz3DError(
            "planar TikZ 3D frame registry must be an object"
        )
    curve = restore_planar_curve_geometry(
        payload,
        expected_curve_id=expected_curve_id,
    )
    if curve.plane_id not in frame_registry:
        raise PlanarTikz3DError(
            f"planar curve references missing registered plane {curve.plane_id!r}"
        )
    raw_frame = frame_registry[curve.plane_id]
    if not isinstance(raw_frame, Mapping):
        raise PlanarTikz3DError(
            f"registered plane {curve.plane_id!r} must contain canonical frame evidence"
        )
    registered = restore_planar_frame_geometry(
        raw_frame,
        expected_plane_id=curve.plane_id,
    )
    if (
        registered.plane_point_names != curve.plane_point_names
        or registered.frame != curve.frame
    ):
        raise PlanarTikz3DError(
            "planar curve frame evidence disagrees with its registered supporting plane"
        )
    return curve


__all__ = [
    "PlanarFrameGeometry3D",
    "PlanarCurveGeometry3D",
    "PlanarTikz3DError",
    "circle_from_plane_coordinates",
    "ellipse_from_plane_coordinates",
    "frame_from_named_points",
    "planar_frame_geometry_payload",
    "planar_curve_geometry_payload",
    "restore_planar_frame_geometry",
    "restore_planar_curve_geometry",
    "restore_registered_planar_curve_geometry",
]
