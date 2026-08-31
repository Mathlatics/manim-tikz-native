"""Static TikZ-facing contracts for certified Dandelin diagrams.

This module deliberately stops before any renderer binding.  It turns named
three-dimensional author data into the existing finite-quadric contracts,
persists redundant canonical evidence, and restores that evidence only by
recomputing it from the caller's authoritative coordinates, plane frame, or
Dandelin construction.

The spatial view keeps curve visibility, teaching-layer order, and physical
surface visibility as three separate claims.  ``depth_aware_diagrammatic``
certifies only analytic stroke intervals, while
``depth_aware_teaching_transparent`` additionally certifies the transparent
teaching-layer order.  Neither mode claims opaque physical surface visibility.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from math import isfinite, radians
import re
import struct
from typing import Literal, TypeAlias

import numpy as np

from polyhedron_visibility.geometry import (
    GeometryContext,
    ResolvedGeometryContext,
)
from polyhedron_visibility.quadrics.contract import (
    ConeModel,
    ConeSpec,
    QuadricContractError,
    SectionPlane,
)
from polyhedron_visibility.quadrics.dandelin import (
    DandelinConstruction3D,
    DandelinConstructionError,
    compute_dandelin_construction,
)
from polyhedron_visibility.quadrics.dandelin_views import (
    DandelinView2DError,
    build_dandelin_meridian_diagram,
    build_dandelin_section_plane_diagram,
)
from polyhedron_visibility.quadrics.planar_curves import PlanarFrame3D


TIKZ_SPACE_RIGHT_CONE_3D_SCHEMA = "tikz-native-space-right-cone-3d/v1"
TIKZ_DANDELIN_CONSTRUCTION_3D_SCHEMA = (
    "tikz-native-dandelin-construction-3d/v1"
)
TIKZ_DANDELIN_STATIC_DIAGRAM_SCHEMA = "tikz-native-dandelin-static-diagram/v3"
TIKZ_DANDELIN_SPATIAL_VIEW_SCHEMA = "tikz-native-dandelin-spatial-view/v1"

DandelinDiagramView: TypeAlias = Literal[
    "spatial",
    "meridian",
    "section-plane",
]
DandelinDiagramMode: TypeAlias = Literal[
    "diagrammatic",
    "depth_aware_diagrammatic",
    "depth_aware_teaching_transparent",
]
GeometryContextInput: TypeAlias = GeometryContext | ResolvedGeometryContext | None

_DANDELIN_DIAGRAM_MODES = frozenset(
    {
        "diagrammatic",
        "depth_aware_diagrammatic",
        "depth_aware_teaching_transparent",
    }
)
_DANDELIN_DEPTH_AWARE_MODES = frozenset(
    {"depth_aware_diagrammatic", "depth_aware_teaching_transparent"}
)

# Authored compiler identities are narrower, but view-local IDs append stable
# role suffixes to those identities.  Keep the contract portable while leaving
# enough room for that deterministic derivation.
_PORTABLE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,511}")
_DIRECTION_RELATIVE_TOLERANCE = float(np.sqrt(np.finfo(float).eps))
_CONE_PAYLOAD_FIELDS = frozenset(
    {
        "schema",
        "coneRef",
        "pointNames",
        "halfAngleDegrees",
        "axialRange",
        "model",
        "cone",
        "static",
    }
)
_CONSTRUCTION_PAYLOAD_FIELDS = frozenset(
    {
        "schema",
        "constructionRef",
        "coneRef",
        "planeRef",
        "construction",
        "static",
    }
)
_DIAGRAM_PAYLOAD_FIELDS = frozenset(
    {
        "schema",
        "diagramId",
        "constructionRef",
        "coneRef",
        "planeRef",
        "view",
        "mode",
        "visibilityAuthoritative",
        "curveVisibilityAuthoritative",
        "surfaceVisibilityAuthoritative",
        "surfaceLayeringAuthoritative",
        "physicalSurfaceVisibilityAuthoritative",
        "static",
        "preset",
        "flags",
        "viewGeometry",
        "semanticObjects",
    }
)
_DIAGRAM_FLAG_FIELDS = frozenset(
    {"showContactCircles", "showFoci", "showDirectrices"}
)


class TikzDandelinContractError(ValueError):
    """Authored Dandelin data is ambiguous, non-canonical, or unsupported."""


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or _PORTABLE_ID.fullmatch(value.strip()) is None:
        raise TikzDandelinContractError(
            f"{label} must be a portable non-empty identifier"
        )
    return value.strip()


def _finite(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TikzDandelinContractError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TikzDandelinContractError(f"{label} must be finite") from exc
    if not isfinite(result):
        raise TikzDandelinContractError(f"{label} must be finite")
    return 0.0 if result == 0.0 else result


def _positive(value: object, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise TikzDandelinContractError(f"{label} must be positive")
    return result


def _point3(value: object, label: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise TikzDandelinContractError(
            f"{label} must contain three finite numbers"
        )
    try:
        raw = tuple(value)  # type: ignore[arg-type]
        if any(isinstance(item, (bool, np.bool_)) for item in raw):
            raise TypeError
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TikzDandelinContractError(
            f"{label} must contain three finite numbers"
        ) from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise TikzDandelinContractError(
            f"{label} must contain three finite numbers"
        )
    return tuple(  # type: ignore[return-value]
        0.0 if item == 0.0 else float(item) for item in result
    )


def _point_names(
    value: object,
    *,
    canonical_payload: bool = False,
) -> tuple[str, str, str]:
    if canonical_payload and not isinstance(value, list):
        raise TikzDandelinContractError(
            "canonical cone pointNames must be a JSON array"
        )
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TikzDandelinContractError(
            "cone point names must contain A/Z/R"
        )
    if len(value) != 3:
        raise TikzDandelinContractError(
            "cone point names must contain exactly A/Z/R"
        )
    names = tuple(
        _identity(item, f"cone point name {index}")
        for index, item in enumerate(value)
    )
    if len(set(names)) != 3:
        raise TikzDandelinContractError(
            "cone point names A/Z/R must be distinct"
        )
    return names  # type: ignore[return-value]


def _axial_range(
    value: object,
    *,
    canonical_payload: bool = False,
) -> tuple[float, float]:
    if canonical_payload and not isinstance(value, list):
        raise TikzDandelinContractError(
            "canonical cone axialRange must be a JSON array"
        )
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise TikzDandelinContractError(
            "cone axial range must contain two finite increasing values"
        )
    lower = _finite(value[0], "cone axial-range minimum")
    upper = _finite(value[1], "cone axial-range maximum")
    if lower >= upper:
        raise TikzDandelinContractError(
            "cone axial range must contain two finite increasing values"
        )
    return lower, upper


def _cone_model(value: object) -> ConeModel:
    if isinstance(value, (bool, np.bool_)):
        raise TikzDandelinContractError("cone model is invalid")
    try:
        model = value if isinstance(value, ConeModel) else ConeModel(value)
    except (TypeError, ValueError) as exc:
        raise TikzDandelinContractError(
            "cone model must be closed_single, open_single, or open_double"
        ) from exc
    if model is ConeModel.ANALYTIC_DOUBLE:
        raise TikzDandelinContractError(
            "analytic_double is not a finite renderable Dandelin cone"
        )
    return model


def _mapping(
    value: object,
    expected_fields: frozenset[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TikzDandelinContractError(f"{label} must be an object")
    keys = frozenset(value.keys())
    if keys != expected_fields:
        missing = sorted(field for field in expected_fields if field not in keys)
        extra = sorted(repr(field) for field in keys if field not in expected_fields)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise TikzDandelinContractError(
            f"{label} fields are invalid: " + "; ".join(details)
        )
    return value  # type: ignore[return-value]


def _strict_json_equal(expected: object, actual: object) -> bool:
    """Compare canonical JSON data without Python's numeric coercions.

    In particular, JSON booleans must never compare equal to numeric ``0`` or
    ``1``, and the sign bit of a floating-point zero remains part of the
    persisted canonical evidence.  Objects may be arbitrary ``Mapping``
    implementations, but arrays must already be JSON-shaped ``list`` values.
    """

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        if any(not isinstance(key, str) for key in actual):
            return False
        if set(expected) != set(actual):
            return False
        return all(
            _strict_json_equal(expected[key], actual[key])
            for key in expected
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(
                _strict_json_equal(expected_item, actual_item)
                for expected_item, actual_item in zip(expected, actual, strict=True)
            )
        )
    if type(expected) is float:
        return type(actual) is float and struct.pack(">d", expected) == struct.pack(
            ">d",
            actual,
        )
    if type(expected) is int:
        return type(actual) is int and expected == actual
    if type(expected) is bool:
        return type(actual) is bool and expected is actual
    if expected is None:
        return actual is None
    if type(expected) is str:
        return type(actual) is str and expected == actual
    return False


def _normalize_canonical_numbers(value: object) -> object:
    """Return JSON-shaped data with every signed zero rewritten as ``0.0``."""

    if isinstance(value, (float, np.floating)):
        result = float(value)
        return 0.0 if result == 0.0 else result
    if isinstance(value, list):
        return [_normalize_canonical_numbers(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_canonical_numbers(item) for item in value]
    if isinstance(value, Mapping):
        return {
            key: _normalize_canonical_numbers(item)
            for key, item in value.items()
        }
    return value


def _scaled_unit(value: np.ndarray, label: str) -> np.ndarray:
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise TikzDandelinContractError(f"{label} must remain finite")
    scale = float(np.max(np.abs(value)))
    if not isfinite(scale) or scale <= 0.0:
        raise TikzDandelinContractError(f"{label} must be non-zero")
    scaled = value / scale
    length = float(np.linalg.norm(scaled))
    if not isfinite(length) or length <= 0.0:
        raise TikzDandelinContractError(f"{label} must be non-zero")
    result = scaled / length
    if not np.all(np.isfinite(result)):
        raise TikzDandelinContractError(f"{label} must remain finite")
    return result


def _named_point(
    coordinates: Mapping[str, object],
    name: str,
) -> tuple[float, float, float]:
    if name not in coordinates:
        raise TikzDandelinContractError(
            f"cone declaration references unknown coordinate {name!r}"
        )
    return _point3(coordinates[name], f"coordinate {name!r}")


def _cone_dict(cone: ConeSpec) -> dict[str, object]:
    return _normalize_canonical_numbers(
        {
            "surfaceId": cone.surface_id,
            "apex": list(cone.apex),
            "axis": list(cone.axis),
            "radialAxis": list(cone.radial_axis or ()),
            "halfAngle": cone.half_angle,
            "axialRange": list(cone.axial_range),
            "model": cone.model.value,
            "componentParentId": cone.component_parent_id,
        }
    )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class SpaceRightCone3DContract:
    """One finite right cone certified from named A/Z/R author points."""

    cone_ref: str
    point_names: tuple[str, str, str]
    half_angle_degrees: float
    axial_range: tuple[float, float]
    model: ConeModel
    cone: ConeSpec
    static: bool = True
    schema: str = TIKZ_SPACE_RIGHT_CONE_3D_SCHEMA

    def __post_init__(self) -> None:
        cone_ref = _identity(self.cone_ref, "cone_ref")
        names = _point_names(self.point_names)
        degrees = _finite(self.half_angle_degrees, "cone half-angle degrees")
        if not 0.0 < degrees < 90.0:
            raise TikzDandelinContractError(
                "cone half-angle degrees must lie strictly between 0 and 90"
            )
        axial = _axial_range(self.axial_range)
        model = _cone_model(self.model)
        if not isinstance(self.cone, ConeSpec):
            raise TypeError("cone must be a ConeSpec")
        if self.static is not True:
            raise TikzDandelinContractError(
                "TikZ Dandelin cone contracts require static=true"
            )
        if self.schema != TIKZ_SPACE_RIGHT_CONE_3D_SCHEMA:
            raise TikzDandelinContractError("invalid TikZ right-cone schema")
        if (
            self.cone.surface_id != cone_ref
            or self.cone.half_angle != radians(degrees)
            or self.cone.axial_range != axial
            or self.cone.model is not model
            or self.cone.component_parent_id is not None
        ):
            raise TikzDandelinContractError(
                "derived cone disagrees with its authored cone contract"
            )
        object.__setattr__(self, "cone_ref", cone_ref)
        object.__setattr__(self, "point_names", names)
        object.__setattr__(self, "half_angle_degrees", degrees)
        object.__setattr__(self, "axial_range", axial)
        object.__setattr__(self, "model", model)

    def to_dict(self) -> dict[str, object]:
        return _normalize_canonical_numbers(
            {
                "schema": self.schema,
                "coneRef": self.cone_ref,
                "pointNames": list(self.point_names),
                "halfAngleDegrees": self.half_angle_degrees,
                "axialRange": list(self.axial_range),
                "model": self.model.value,
                "cone": _cone_dict(self.cone),
                "static": True,
            }
        )  # type: ignore[return-value]

    def canonical_json(self) -> str:
        return canonical_dandelin_contract_json(self)


def build_space_right_cone_contract(
    cone_ref: str,
    point_names: Sequence[str],
    coordinates: Mapping[str, object],
    half_angle_degrees: object,
    axial_range: Sequence[object],
    model: ConeModel | str,
) -> SpaceRightCone3DContract:
    """Build a finite :class:`ConeSpec` from named ``A/Z/R`` coordinates.

    ``A`` is the apex, ``Z-A`` determines the positive cone axis, and the
    component of ``R-A`` orthogonal to that axis determines the positive
    radial direction.  The two directions are normalized with scale-aware
    arithmetic before constructing the core quadric contract.
    """

    identity = _identity(cone_ref, "cone_ref")
    names = _point_names(point_names)
    if not isinstance(coordinates, Mapping):
        raise TikzDandelinContractError("named coordinates must be an object")
    apex = np.asarray(_named_point(coordinates, names[0]), dtype=float)
    axis_point = np.asarray(_named_point(coordinates, names[1]), dtype=float)
    radial_point = np.asarray(_named_point(coordinates, names[2]), dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        axis_delta = axis_point - apex
        radial_delta = radial_point - apex
    axis = _scaled_unit(axis_delta, "A-to-Z cone axis")
    radial_seed = _scaled_unit(radial_delta, "A-to-R radial seed")
    with np.errstate(over="ignore", invalid="ignore"):
        perpendicular = radial_seed - float(np.dot(radial_seed, axis)) * axis
    perpendicular_length = float(np.linalg.norm(perpendicular))
    if (
        not isfinite(perpendicular_length)
        or perpendicular_length <= _DIRECTION_RELATIVE_TOLERANCE
    ):
        raise TikzDandelinContractError(
            "A/Z/R must define a radial direction distinguishable from the cone axis"
        )
    radial = _scaled_unit(perpendicular, "projected cone radial direction")
    degrees = _finite(half_angle_degrees, "cone half-angle degrees")
    if not 0.0 < degrees < 90.0:
        raise TikzDandelinContractError(
            "cone half-angle degrees must lie strictly between 0 and 90"
        )
    axial = _axial_range(axial_range)
    resolved_model = _cone_model(model)
    try:
        cone = ConeSpec(
            identity,
            tuple(float(item) for item in apex),
            tuple(float(item) for item in axis),
            radians(degrees),
            axial,
            radial_axis=tuple(float(item) for item in radial),
            model=resolved_model,
        )
    except (QuadricContractError, TypeError, ValueError, OverflowError) as exc:
        raise TikzDandelinContractError(
            f"finite right cone cannot be certified: {exc}"
        ) from exc
    return SpaceRightCone3DContract(
        identity,
        names,
        degrees,
        axial,
        resolved_model,
        cone,
    )


def restore_space_right_cone_contract(
    payload: Mapping[str, object],
    coordinates: Mapping[str, object],
    *,
    expected_cone_ref: str | None = None,
) -> SpaceRightCone3DContract:
    """Recompute one cone from authoritative named coordinates and compare."""

    raw = _mapping(payload, _CONE_PAYLOAD_FIELDS, "right-cone payload")
    if raw["schema"] != TIKZ_SPACE_RIGHT_CONE_3D_SCHEMA:
        raise TikzDandelinContractError("invalid TikZ right-cone schema")
    if raw["static"] is not True:
        raise TikzDandelinContractError(
            "TikZ Dandelin cone payload requires static=true"
        )
    if not isinstance(raw["cone"], Mapping):
        raise TikzDandelinContractError("canonical derived cone must be an object")
    cone_ref = _identity(raw["coneRef"], "coneRef")
    if expected_cone_ref is not None and cone_ref != _identity(
        expected_cone_ref,
        "expected_cone_ref",
    ):
        raise TikzDandelinContractError(
            "restored cone identity disagrees with its registry identity"
        )
    result = build_space_right_cone_contract(
        cone_ref,
        _point_names(raw["pointNames"], canonical_payload=True),
        coordinates,
        raw["halfAngleDegrees"],
        _axial_range(raw["axialRange"], canonical_payload=True),
        _cone_model(raw["model"]),
    )
    if not _strict_json_equal(result.to_dict(), raw):
        raise TikzDandelinContractError(
            "right-cone payload is stale, forged, or non-canonical"
        )
    return result


def section_plane_from_planar_frame(
    frame: PlanarFrame3D,
    *,
    expected_plane_ref: str | None = None,
) -> SectionPlane:
    """Restore a mathematical section plane from one certified static frame."""

    if not isinstance(frame, PlanarFrame3D):
        raise TypeError("frame must be a PlanarFrame3D")
    plane_ref = frame.frame_id
    if expected_plane_ref is not None and plane_ref != _identity(
        expected_plane_ref,
        "expected_plane_ref",
    ):
        raise TikzDandelinContractError(
            "section-plane frame identity disagrees with planeRef"
        )
    try:
        plane = SectionPlane(
            plane_ref,
            frame.point,
            frame.normal,
            u_axis=frame.u_axis,
        )
    except (QuadricContractError, TypeError, ValueError, OverflowError) as exc:
        raise TikzDandelinContractError(
            f"section plane cannot be restored from its certified frame: {exc}"
        ) from exc
    basis_error = max(
        float(
            np.max(
                np.abs(
                    np.asarray(actual, dtype=float)
                    - np.asarray(expected, dtype=float)
                )
            )
        )
        for actual, expected in (
            (plane.normal, frame.normal),
            (plane.u_axis, frame.u_axis),
        )
    )
    if (
        plane.point != frame.point
        or not isfinite(basis_error)
        or basis_error > 64.0 * np.finfo(float).eps
    ):
        raise TikzDandelinContractError(
            "section plane changed the authoritative planar-frame basis"
        )
    return plane


@dataclass(frozen=True, slots=True)
class DandelinConstructionContract:
    """Canonical static registration for one finite Dandelin construction."""

    construction_ref: str
    cone_ref: str
    plane_ref: str
    construction: DandelinConstruction3D
    static: bool = True
    schema: str = TIKZ_DANDELIN_CONSTRUCTION_3D_SCHEMA

    def __post_init__(self) -> None:
        construction_ref = _identity(self.construction_ref, "construction_ref")
        cone_ref = _identity(self.cone_ref, "cone_ref")
        plane_ref = _identity(self.plane_ref, "plane_ref")
        if not isinstance(self.construction, DandelinConstruction3D):
            raise TypeError("construction must be a DandelinConstruction3D")
        if self.static is not True:
            raise TikzDandelinContractError(
                "TikZ Dandelin constructions require static=true"
            )
        if self.schema != TIKZ_DANDELIN_CONSTRUCTION_3D_SCHEMA:
            raise TikzDandelinContractError(
                "invalid TikZ Dandelin-construction schema"
            )
        if (
            self.construction.construction_id != construction_ref
            or self.construction.cone.surface_id != cone_ref
            or self.construction.plane.plane_id != plane_ref
            or self.construction.section_frame.frame_id
            != f"{construction_ref}:section-plane"
        ):
            raise TikzDandelinContractError(
                "Dandelin construction disagrees with its registry references"
            )
        object.__setattr__(self, "construction_ref", construction_ref)
        object.__setattr__(self, "cone_ref", cone_ref)
        object.__setattr__(self, "plane_ref", plane_ref)

    def to_dict(self) -> dict[str, object]:
        return _normalize_canonical_numbers(
            {
                "schema": self.schema,
                "constructionRef": self.construction_ref,
                "coneRef": self.cone_ref,
                "planeRef": self.plane_ref,
                "construction": self.construction.to_dict(),
                "static": True,
            }
        )  # type: ignore[return-value]

    def canonical_json(self) -> str:
        return canonical_dandelin_contract_json(self)


def build_dandelin_construction_contract(
    construction_ref: str,
    *,
    cone_ref: str,
    cone: ConeSpec,
    plane_ref: str,
    plane_frame: PlanarFrame3D,
    context: GeometryContextInput = None,
    coefficient_tolerance: float | None = None,
) -> DandelinConstructionContract:
    """Compute and persist one construction from authoritative cone and plane."""

    identity = _identity(construction_ref, "construction_ref")
    resolved_cone_ref = _identity(cone_ref, "cone_ref")
    resolved_plane_ref = _identity(plane_ref, "plane_ref")
    if not isinstance(cone, ConeSpec):
        raise TypeError("cone must be a ConeSpec")
    if cone.surface_id != resolved_cone_ref:
        raise TikzDandelinContractError(
            "authoritative cone identity disagrees with coneRef"
        )
    plane = section_plane_from_planar_frame(
        plane_frame,
        expected_plane_ref=resolved_plane_ref,
    )
    resolved_coefficient_tolerance = (
        None
        if coefficient_tolerance is None
        else _positive(coefficient_tolerance, "coefficient_tolerance")
    )
    try:
        construction = compute_dandelin_construction(
            identity,
            cone,
            plane,
            context=context,
            coefficient_tolerance=resolved_coefficient_tolerance,
        )
    except (
        DandelinConstructionError,
        QuadricContractError,
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise TikzDandelinContractError(
            f"Dandelin construction cannot be certified: {exc}"
        ) from exc
    return DandelinConstructionContract(
        identity,
        resolved_cone_ref,
        resolved_plane_ref,
        construction,
    )


def restore_dandelin_construction_contract(
    payload: Mapping[str, object],
    *,
    cone: ConeSpec,
    plane_frame: PlanarFrame3D,
    context: GeometryContextInput = None,
    coefficient_tolerance: float | None = None,
    expected_construction_ref: str | None = None,
    expected_cone_ref: str | None = None,
    expected_plane_ref: str | None = None,
) -> DandelinConstructionContract:
    """Recompute a persisted construction from its authoritative dependencies."""

    raw = _mapping(
        payload,
        _CONSTRUCTION_PAYLOAD_FIELDS,
        "Dandelin-construction payload",
    )
    if raw["schema"] != TIKZ_DANDELIN_CONSTRUCTION_3D_SCHEMA:
        raise TikzDandelinContractError(
            "invalid TikZ Dandelin-construction schema"
        )
    if raw["static"] is not True:
        raise TikzDandelinContractError(
            "TikZ Dandelin-construction payload requires static=true"
        )
    if not isinstance(raw["construction"], Mapping):
        raise TikzDandelinContractError(
            "canonical Dandelin construction must be an object"
        )
    construction_ref = _identity(raw["constructionRef"], "constructionRef")
    cone_ref = _identity(raw["coneRef"], "coneRef")
    plane_ref = _identity(raw["planeRef"], "planeRef")
    expected = (
        (construction_ref, expected_construction_ref, "construction"),
        (cone_ref, expected_cone_ref, "cone"),
        (plane_ref, expected_plane_ref, "plane"),
    )
    for actual, requested, label in expected:
        if requested is not None and actual != _identity(
            requested,
            f"expected_{label}_ref",
        ):
            raise TikzDandelinContractError(
                f"restored {label} identity disagrees with its registry identity"
            )
    result = build_dandelin_construction_contract(
        construction_ref,
        cone_ref=cone_ref,
        cone=cone,
        plane_ref=plane_ref,
        plane_frame=plane_frame,
        context=context,
        coefficient_tolerance=coefficient_tolerance,
    )
    if not _strict_json_equal(result.to_dict(), raw):
        raise TikzDandelinContractError(
            "Dandelin-construction payload is stale, forged, or non-canonical"
        )
    return result


@dataclass(frozen=True, slots=True)
class DandelinSemanticObject:
    """One view-local drawable identity mapped to shared source geometry."""

    object_id: str
    role: str
    source_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", _identity(self.object_id, "object_id"))
        object.__setattr__(self, "role", _identity(self.role, "semantic role"))
        object.__setattr__(self, "source_ref", _identity(self.source_ref, "source_ref"))

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.object_id,
            "role": self.role,
            "sourceRef": self.source_ref,
        }


def _diagram_view(value: object) -> DandelinDiagramView:
    if value not in {"spatial", "meridian", "section-plane"}:
        raise TikzDandelinContractError(
            "Dandelin diagram view must be spatial, meridian, or section-plane"
        )
    return value  # type: ignore[return-value]


def _resolved_flag(value: object, default: bool, label: str) -> bool:
    if value is None:
        return default
    if type(value) is not bool:
        raise TikzDandelinContractError(f"{label} must be a boolean")
    return value


def _diagram_id(construction: DandelinConstruction3D, view: str) -> str:
    return f"{construction.construction_id}:view:{view}"


def _diagram_view_geometry(
    construction: DandelinConstruction3D,
    view: DandelinDiagramView,
) -> dict[str, object]:
    try:
        if view == "spatial":
            value: object = {
                "schema": TIKZ_DANDELIN_SPATIAL_VIEW_SCHEMA,
                "construction": construction.to_dict(),
            }
        elif view == "meridian":
            value = build_dandelin_meridian_diagram(construction).to_dict()
        else:
            value = build_dandelin_section_plane_diagram(construction).to_dict()
    except (DandelinView2DError, DandelinConstructionError, ValueError) as exc:
        raise TikzDandelinContractError(
            f"Dandelin {view} view cannot be certified: {exc}"
        ) from exc
    return _normalize_canonical_numbers(value)  # type: ignore[return-value]


def build_dandelin_semantic_plan(
    construction: DandelinConstruction3D,
    view: DandelinDiagramView | str,
    *,
    show_contact_circles: bool,
    show_foci: bool,
    show_directrices: bool,
) -> tuple[DandelinSemanticObject, ...]:
    """Return the one-to-one logical object plan used by contract and renderer."""

    if not isinstance(construction, DandelinConstruction3D):
        raise TypeError("construction must be a DandelinConstruction3D")
    resolved_view = _diagram_view(view)
    for value, label in (
        (show_contact_circles, "show_contact_circles"),
        (show_foci, "show_foci"),
        (show_directrices, "show_directrices"),
    ):
        if type(value) is not bool:
            raise TikzDandelinContractError(f"{label} must be a boolean")
    if resolved_view == "section-plane" and show_contact_circles:
        raise TikzDandelinContractError(
            "section-plane view cannot display sphere/contact circles; "
            "they are not geometry in the cutting plane"
        )
    if resolved_view == "meridian" and show_directrices:
        raise TikzDandelinContractError(
            "meridian view cannot display section-plane directrix lines"
        )

    view = resolved_view
    diagram_id = _diagram_id(construction, view)
    records: list[DandelinSemanticObject] = []
    role_counts: dict[str, int] = {}

    def add(role: str, source_ref: str) -> None:
        index = role_counts.get(role, 0)
        role_counts[role] = index + 1
        records.append(
            DandelinSemanticObject(
                f"{diagram_id}:object:{role}:{index:04d}",
                role,
                source_ref,
            )
        )

    if view == "spatial":
        add("cone_surface", construction.cone.surface_id)
        add("section_plane", construction.plane.plane_id)
        for sphere in construction.spheres:
            add("sphere_surface", sphere.sphere_id)
        add("section_curve", f"{construction.construction_id}:section")
        if show_contact_circles:
            for curve in construction.cone_contact_circles:
                add("contact_circle", curve.curve_id)
        if show_directrices:
            for directrix in construction.directrices:
                add("directrix", directrix.directrix_id)
        if show_foci:
            for sphere in construction.spheres:
                add("focus", sphere.focus_id)
        return tuple(records)

    if view == "meridian":
        diagram = build_dandelin_meridian_diagram(construction)
        for nappe in ("negative", "positive"):
            generators = tuple(
                item
                for item in diagram.generators
                if f":nappe:{nappe}:" in item.segment_id
            )
            if not generators:
                continue
            add("cone_face", construction.cone.surface_id)
            for generator in generators:
                add("cone_generator", generator.source_ref)
        add("section_line", diagram.section_line.source_ref)
        for circle in diagram.sphere_circles:
            add("sphere_circle_section", circle.source_ref)
        if show_contact_circles:
            for tangency in diagram.tangencies:
                if tangency.carrier_id != diagram.section_line.line_id:
                    add("contact_circle_section_point", tangency.source_ref)
        if show_foci:
            for focus in diagram.focus_points:
                add("focus", focus.source_ref)
        return tuple(records)

    diagram = build_dandelin_section_plane_diagram(construction)
    add("section_curve", f"{construction.construction_id}:section")
    if show_directrices:
        for directrix in diagram.directrices:
            add("directrix", directrix.source_ref)
    if show_foci:
        for focus in diagram.focus_points:
            add("focus", focus.source_ref)
    return tuple(records)


@dataclass(frozen=True, slots=True)
class DandelinStaticDiagramContract:
    """One immutable classroom view over an authoritative construction."""

    construction: DandelinConstruction3D
    view: DandelinDiagramView
    show_contact_circles: bool
    show_foci: bool
    show_directrices: bool
    view_geometry: Mapping[str, object]
    semantic_objects: tuple[DandelinSemanticObject, ...]
    preset: str = "classroom"
    mode: str = "diagrammatic"
    visibility_authoritative: bool = False
    static: bool = True
    schema: str = TIKZ_DANDELIN_STATIC_DIAGRAM_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.construction, DandelinConstruction3D):
            raise TypeError("construction must be a DandelinConstruction3D")
        view = _diagram_view(self.view)
        if self.preset != "classroom":
            raise TikzDandelinContractError(
                "Dandelin static diagram supports only preset=classroom"
            )
        if self.mode not in _DANDELIN_DIAGRAM_MODES:
            raise TikzDandelinContractError(
                "Dandelin static diagram mode must be diagrammatic, "
                "depth_aware_diagrammatic, or "
                "depth_aware_teaching_transparent"
            )
        if self.mode in _DANDELIN_DEPTH_AWARE_MODES and view != "spatial":
            raise TikzDandelinContractError(
                f"{self.mode} mode is only valid for the spatial view"
            )
        if self.visibility_authoritative is not False:
            raise TikzDandelinContractError(
                "Dandelin static diagrams are not visibility-authoritative"
            )
        if self.static is not True:
            raise TikzDandelinContractError(
                "Dandelin static diagram payload requires static=true"
            )
        if self.schema != TIKZ_DANDELIN_STATIC_DIAGRAM_SCHEMA:
            raise TikzDandelinContractError("invalid Dandelin static-diagram schema")
        for name in (
            "show_contact_circles",
            "show_foci",
            "show_directrices",
        ):
            if type(getattr(self, name)) is not bool:
                raise TikzDandelinContractError(f"{name} must be a boolean")
        if (
            self.mode == "depth_aware_teaching_transparent"
            and not self.show_contact_circles
        ):
            raise TikzDandelinContractError(
                "depth_aware_teaching_transparent requires "
                "show_contact_circles=true because those strokes own the "
                "certified equal-depth seams"
            )
        if view == "section-plane" and self.show_contact_circles:
            raise TikzDandelinContractError(
                "section-plane view cannot display sphere/contact circles; "
                "they are not geometry in the cutting plane"
            )
        if view == "meridian" and self.show_directrices:
            raise TikzDandelinContractError(
                "meridian view cannot display section-plane directrix lines"
            )
        expected_geometry = _diagram_view_geometry(self.construction, view)
        if not isinstance(self.view_geometry, Mapping) or not _strict_json_equal(
            expected_geometry,
            self.view_geometry,
        ):
            raise TikzDandelinContractError(
                "diagram view geometry does not match its authoritative construction"
            )
        expected_objects = build_dandelin_semantic_plan(
            self.construction,
            view,
            show_contact_circles=self.show_contact_circles,
            show_foci=self.show_foci,
            show_directrices=self.show_directrices,
        )
        if tuple(self.semantic_objects) != expected_objects:
            raise TikzDandelinContractError(
                "semanticObjects do not match the canonical diagram view"
            )
        object.__setattr__(self, "view", view)
        object.__setattr__(self, "view_geometry", expected_geometry)
        object.__setattr__(self, "semantic_objects", expected_objects)

    @property
    def diagram_id(self) -> str:
        return _diagram_id(self.construction, self.view)

    @property
    def construction_ref(self) -> str:
        return self.construction.construction_id

    @property
    def cone_ref(self) -> str:
        return self.construction.cone.surface_id

    @property
    def plane_ref(self) -> str:
        return self.construction.plane.plane_id

    @property
    def curve_visibility_authoritative(self) -> bool:
        return self.mode in _DANDELIN_DEPTH_AWARE_MODES

    @property
    def surface_visibility_authoritative(self) -> bool:
        return False

    @property
    def surface_layering_authoritative(self) -> bool:
        return self.mode == "depth_aware_teaching_transparent"

    @property
    def physical_surface_visibility_authoritative(self) -> bool:
        return False

    def to_dict(self) -> dict[str, object]:
        return _normalize_canonical_numbers(
            {
                "schema": self.schema,
                "diagramId": self.diagram_id,
                "constructionRef": self.construction_ref,
                "coneRef": self.cone_ref,
                "planeRef": self.plane_ref,
                "view": self.view,
                "mode": self.mode,
                "visibilityAuthoritative": False,
                "curveVisibilityAuthoritative": (
                    self.curve_visibility_authoritative
                ),
                "surfaceVisibilityAuthoritative": False,
                "surfaceLayeringAuthoritative": (
                    self.surface_layering_authoritative
                ),
                "physicalSurfaceVisibilityAuthoritative": False,
                "static": True,
                "preset": "classroom",
                "flags": {
                    "showContactCircles": self.show_contact_circles,
                    "showFoci": self.show_foci,
                    "showDirectrices": self.show_directrices,
                },
                "viewGeometry": dict(self.view_geometry),
                "semanticObjects": [
                    item.to_dict() for item in self.semantic_objects
                ],
            }
        )  # type: ignore[return-value]

    def canonical_json(self) -> str:
        return canonical_dandelin_contract_json(self)


_VIEW_FLAG_DEFAULTS: dict[DandelinDiagramView, tuple[bool, bool, bool]] = {
    "spatial": (True, True, True),
    "meridian": (True, True, False),
    "section-plane": (False, True, True),
}


def build_dandelin_static_diagram_contract(
    construction: DandelinConstruction3D,
    *,
    view: DandelinDiagramView | str,
    preset: str = "classroom",
    mode: DandelinDiagramMode | str = "diagrammatic",
    show_contact_circles: bool | None = None,
    show_foci: bool | None = None,
    show_directrices: bool | None = None,
) -> DandelinStaticDiagramContract:
    """Build one static spatial, meridian, or cutting-plane diagram payload."""

    if not isinstance(construction, DandelinConstruction3D):
        raise TypeError("construction must be a DandelinConstruction3D")
    resolved_view = _diagram_view(view)
    if preset != "classroom":
        raise TikzDandelinContractError(
            "Dandelin static diagram supports only preset=classroom"
        )
    if not isinstance(mode, str) or mode not in _DANDELIN_DIAGRAM_MODES:
        raise TikzDandelinContractError(
            "Dandelin static diagram mode must be diagrammatic, "
            "depth_aware_diagrammatic, or "
            "depth_aware_teaching_transparent"
        )
    if mode in _DANDELIN_DEPTH_AWARE_MODES and resolved_view != "spatial":
        raise TikzDandelinContractError(
            f"{mode} mode is only valid for the spatial view"
        )
    defaults = _VIEW_FLAG_DEFAULTS[resolved_view]
    contact = _resolved_flag(
        show_contact_circles,
        defaults[0],
        "show_contact_circles",
    )
    foci = _resolved_flag(show_foci, defaults[1], "show_foci")
    directrices = _resolved_flag(
        show_directrices,
        defaults[2],
        "show_directrices",
    )
    if mode == "depth_aware_teaching_transparent" and not contact:
        raise TikzDandelinContractError(
            "depth_aware_teaching_transparent requires "
            "show_contact_circles=true because those strokes own the "
            "certified equal-depth seams"
        )
    if resolved_view == "section-plane" and contact:
        raise TikzDandelinContractError(
            "section-plane view cannot display sphere/contact circles; "
            "they are not geometry in the cutting plane"
        )
    if resolved_view == "meridian" and directrices:
        raise TikzDandelinContractError(
            "meridian view cannot display section-plane directrix lines"
        )
    geometry = _diagram_view_geometry(construction, resolved_view)
    semantic = build_dandelin_semantic_plan(
        construction,
        resolved_view,
        show_contact_circles=contact,
        show_foci=foci,
        show_directrices=directrices,
    )
    return DandelinStaticDiagramContract(
        construction,
        resolved_view,
        contact,
        foci,
        directrices,
        geometry,
        semantic,
        mode=mode,
    )


def restore_dandelin_static_diagram_contract(
    payload: Mapping[str, object],
    construction: DandelinConstruction3D,
    *,
    expected_diagram_id: str | None = None,
) -> DandelinStaticDiagramContract:
    """Recompute a static diagram from its authoritative construction."""

    raw = _mapping(payload, _DIAGRAM_PAYLOAD_FIELDS, "Dandelin-diagram payload")
    if raw["schema"] != TIKZ_DANDELIN_STATIC_DIAGRAM_SCHEMA:
        raise TikzDandelinContractError("invalid Dandelin static-diagram schema")
    if (
        not isinstance(raw["mode"], str)
        or raw["mode"] not in _DANDELIN_DIAGRAM_MODES
    ):
        raise TikzDandelinContractError(
            "Dandelin static diagram mode must be diagrammatic, "
            "depth_aware_diagrammatic, or "
            "depth_aware_teaching_transparent"
        )
    if raw["visibilityAuthoritative"] is not False:
        raise TikzDandelinContractError(
            "Dandelin static diagrams are not visibility-authoritative"
        )
    expected_curve_authority = raw["mode"] in _DANDELIN_DEPTH_AWARE_MODES
    if raw["curveVisibilityAuthoritative"] is not expected_curve_authority:
        raise TikzDandelinContractError(
            "curveVisibilityAuthoritative disagrees with Dandelin diagram mode"
        )
    if raw["surfaceVisibilityAuthoritative"] is not False:
        raise TikzDandelinContractError(
            "surfaceVisibilityAuthoritative is reserved for unsupported "
            "physical surface visibility"
        )
    expected_layer_authority = (
        raw["mode"] == "depth_aware_teaching_transparent"
    )
    if raw["surfaceLayeringAuthoritative"] is not expected_layer_authority:
        raise TikzDandelinContractError(
            "surfaceLayeringAuthoritative disagrees with Dandelin diagram mode"
        )
    if raw["physicalSurfaceVisibilityAuthoritative"] is not False:
        raise TikzDandelinContractError(
            "Dandelin diagrams are not physical-surface-visibility-authoritative"
        )
    if raw["static"] is not True:
        raise TikzDandelinContractError(
            "Dandelin static diagram payload requires static=true"
        )
    if raw["preset"] != "classroom":
        raise TikzDandelinContractError(
            "Dandelin static diagram supports only preset=classroom"
        )
    flags = _mapping(raw["flags"], _DIAGRAM_FLAG_FIELDS, "diagram flags")
    for name in _DIAGRAM_FLAG_FIELDS:
        if type(flags[name]) is not bool:
            raise TikzDandelinContractError(f"diagram flag {name} must be a boolean")
    if not isinstance(raw["viewGeometry"], Mapping):
        raise TikzDandelinContractError("viewGeometry must be an object")
    if not isinstance(raw["semanticObjects"], list) or any(
        not isinstance(item, Mapping) for item in raw["semanticObjects"]
    ):
        raise TikzDandelinContractError("semanticObjects must be a JSON array of objects")
    diagram_id = _identity(raw["diagramId"], "diagramId")
    if expected_diagram_id is not None and diagram_id != _identity(
        expected_diagram_id,
        "expected_diagram_id",
    ):
        raise TikzDandelinContractError(
            "restored diagram identity disagrees with its object identity"
        )
    for field, actual in (
        ("constructionRef", construction.construction_id),
        ("coneRef", construction.cone.surface_id),
        ("planeRef", construction.plane.plane_id),
    ):
        if _identity(raw[field], field) != actual:
            raise TikzDandelinContractError(
                f"{field} disagrees with the authoritative construction"
            )
    result = build_dandelin_static_diagram_contract(
        construction,
        view=_diagram_view(raw["view"]),
        preset="classroom",
        mode=str(raw["mode"]),
        show_contact_circles=flags["showContactCircles"],  # type: ignore[arg-type]
        show_foci=flags["showFoci"],  # type: ignore[arg-type]
        show_directrices=flags["showDirectrices"],  # type: ignore[arg-type]
    )
    if result.diagram_id != diagram_id or not _strict_json_equal(
        result.to_dict(),
        raw,
    ):
        raise TikzDandelinContractError(
            "Dandelin-diagram payload is stale, forged, or non-canonical"
        )
    return result


DandelinContractValue: TypeAlias = (
    SpaceRightCone3DContract
    | DandelinConstructionContract
    | DandelinStaticDiagramContract
    | Mapping[str, object]
)


def canonical_dandelin_contract_json(value: DandelinContractValue) -> str:
    """Encode any contract or payload deterministically without signed zero."""

    if isinstance(
        value,
        (
            SpaceRightCone3DContract,
            DandelinConstructionContract,
            DandelinStaticDiagramContract,
        ),
    ):
        payload: object = value.to_dict()
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError("value must be a Dandelin contract or payload mapping")
    try:
        return json.dumps(
            _normalize_canonical_numbers(payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise TikzDandelinContractError(
            f"Dandelin payload is not canonical JSON data: {exc}"
        ) from exc


__all__ = [
    "DandelinConstructionContract",
    "DandelinContractValue",
    "DandelinDiagramView",
    "DandelinDiagramMode",
    "DandelinSemanticObject",
    "DandelinStaticDiagramContract",
    "SpaceRightCone3DContract",
    "TIKZ_DANDELIN_CONSTRUCTION_3D_SCHEMA",
    "TIKZ_DANDELIN_SPATIAL_VIEW_SCHEMA",
    "TIKZ_DANDELIN_STATIC_DIAGRAM_SCHEMA",
    "TIKZ_SPACE_RIGHT_CONE_3D_SCHEMA",
    "TikzDandelinContractError",
    "build_dandelin_construction_contract",
    "build_dandelin_semantic_plan",
    "build_dandelin_static_diagram_contract",
    "build_space_right_cone_contract",
    "canonical_dandelin_contract_json",
    "restore_dandelin_construction_contract",
    "restore_dandelin_static_diagram_contract",
    "restore_space_right_cone_contract",
    "section_plane_from_planar_frame",
]
