"""Certified parent layers for a mother surface with tangent inner spheres.

The global quadric compositor deliberately rejects touching or nested solids.
That remains the correct fail-closed behaviour for an unregistered scene.  A
small class of teaching diagrams, notably a Dandelin construction, carries
stronger authored evidence: every inner sphere is tangent to one mother
surface along a named analytic circle and tangent to the displayed cutting
plane at one point.

This module validates that evidence and inserts the sphere bodies into the
already-certified mother/plane painter graph.  It does not solve curve
visibility; :mod:`scene_occlusion` attaches the shared boundary compositor
after this parent graph has been formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, sin
from typing import Mapping, Sequence

import numpy as np

from ..compositor import PainterConstraint, stable_topological_sort
from ..geometry import GeometryQuantity, ResolvedGeometryContext
from ..parallel_solver import ParallelView
from .boundary_compositing import (
    BoundaryOcclusionScope,
    QuadricBoundarySource,
)
from .compositing import QuadricPaintPolicy, QuadricPaintRelation
from .contract import ConeSpec, CylinderSpec, SectionPlane, SphereSpec
from .curves import EllipseArcCurve
from .global_occlusion import GlobalQuadricFrame, compute_global_quadric_frame
from .section_compositing import PlaneDepthRole, QuadricSectionCompositingFrame


NESTED_TANGENT_PARENT_FRAME_SCHEMA = "manim-nested-tangent-parent-frame/v1"
NESTED_TANGENT_CONTACT_EVIDENCE_SCHEMA = (
    "manim-nested-tangent-contact-evidence/v1"
)


class NestedTangentCompositingError(ValueError):
    """A registered nested/tangent parent graph cannot be certified."""


class TangentSpherePlanePosition(str, Enum):
    """Depth position of a sphere relative to its tangent cutting plane."""

    PLANE_IN_FRONT = "plane_in_front"
    SPHERE_IN_FRONT = "sphere_in_front"


def _identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NestedTangentCompositingError(
            f"{label} must be a non-empty string"
        )
    return value.strip()


def _local_boundary_tolerance(
    context: ResolvedGeometryContext,
    scale: float,
    conditioning_values: Sequence[float] = (),
) -> float:
    """Resolve a boundary tolerance without importing remote scene extent.

    An explicit boundary override is an authored contract and remains
    authoritative.  Otherwise the relative policy is resolved against the
    local contact scale.  A small ULP allowance covers subtraction at large
    translated coordinates without letting an unrelated trim endpoint relax
    a local tangency certificate.
    """

    explicit = context.overrides.get(GeometryQuantity.BOUNDARY)
    local_scale = max(float(scale), context.policy.absolute_floor)
    policy_tolerance = (
        float(explicit)
        if explicit is not None
        else context.policy.boundary_factor
        * max(
            context.policy.absolute_floor,
            context.policy.relative * local_scale,
        )
    )
    conditioning_scale = max(
        (abs(float(value)) for value in conditioning_values),
        default=local_scale,
    )
    ulp_tolerance = 32.0 * abs(float(np.spacing(conditioning_scale)))
    return max(policy_tolerance, ulp_tolerance)


def _point_pair_tolerance(
    context: ResolvedGeometryContext,
    scale: float,
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    """Tolerance for subtracting two nearby authored world points."""

    local = _local_boundary_tolerance(context, scale)
    conditioning_scale = max(
        *(abs(float(value)) for value in first),
        *(abs(float(value)) for value in second),
        float(scale),
    )
    return max(
        local,
        8.0 * abs(float(np.spacing(conditioning_scale))),
    )


@dataclass(frozen=True, slots=True)
class NestedTangentSphereSpec:
    """Bind one sphere body to its mother and analytic contact-circle source."""

    sphere_surface_id: str
    mother_surface_id: str
    contact_source_id: str
    sphere_item_id: str

    def __post_init__(self) -> None:
        values = {
            name: _identity(getattr(self, name), name)
            for name in self.__dataclass_fields__
        }
        if values["sphere_surface_id"] == values["mother_surface_id"]:
            raise NestedTangentCompositingError(
                "a nested sphere cannot be its own mother surface"
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class NestedTangentContactEvidence:
    """Validated mother/sphere/plane contact evidence for one sphere."""

    sphere_surface_id: str
    mother_surface_id: str
    contact_source_id: str
    sphere_item_id: str
    plane_signed_distance: float
    plane_ray_parameter: float
    plane_position: TangentSpherePlanePosition
    max_sphere_contact_residual: float
    max_mother_contact_residual: float
    max_normal_cross_residual: float
    schema: str = NESTED_TANGENT_CONTACT_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NESTED_TANGENT_CONTACT_EVIDENCE_SCHEMA:
            raise NestedTangentCompositingError(
                "invalid nested-tangent contact evidence schema"
            )
        for name in (
            "sphere_surface_id",
            "mother_surface_id",
            "contact_source_id",
            "sphere_item_id",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        for name in (
            "plane_signed_distance",
            "plane_ray_parameter",
            "max_sphere_contact_residual",
            "max_mother_contact_residual",
            "max_normal_cross_residual",
        ):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise NestedTangentCompositingError(
                    f"{name} must be finite"
                )
            if name.startswith("max_") and value < 0.0:
                raise NestedTangentCompositingError(
                    f"{name} must be non-negative"
                )
            object.__setattr__(self, name, value)
        if self.plane_ray_parameter == 0.0:
            raise NestedTangentCompositingError(
                "a tangent sphere cannot have unresolved plane depth"
            )
        if not isinstance(self.plane_position, TangentSpherePlanePosition):
            raise TypeError(
                "plane_position must be a TangentSpherePlanePosition"
            )
        expected = (
            TangentSpherePlanePosition.PLANE_IN_FRONT
            if self.plane_ray_parameter > 0.0
            else TangentSpherePlanePosition.SPHERE_IN_FRONT
        )
        if self.plane_position is not expected:
            raise NestedTangentCompositingError(
                "plane position disagrees with the certified ray parameter"
            )

    @property
    def plane_is_in_front(self) -> bool:
        return self.plane_position is TangentSpherePlanePosition.PLANE_IN_FRONT

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sphereSurfaceId": self.sphere_surface_id,
            "motherSurfaceId": self.mother_surface_id,
            "contactSourceId": self.contact_source_id,
            "sphereItemId": self.sphere_item_id,
            "planeSignedDistance": self.plane_signed_distance,
            "planeRayParameter": self.plane_ray_parameter,
            "planePosition": self.plane_position.value,
            "maxSphereContactResidual": self.max_sphere_contact_residual,
            "maxMotherContactResidual": self.max_mother_contact_residual,
            "maxNormalCrossResidual": self.max_normal_cross_residual,
        }


@dataclass(frozen=True, slots=True)
class NestedTangentParentFrame:
    """One complete parent painter graph before semantic curves are attached."""

    mother_surface_id: str
    contacts: tuple[NestedTangentContactEvidence, ...]
    sphere_pair_frame: GlobalQuadricFrame
    parent_item_ids: tuple[str, ...]
    order_relations: tuple[QuadricPaintRelation, ...]
    draw_order: tuple[str, ...]
    surface_item_by_id: tuple[tuple[str, str], ...]
    surface_layering_authoritative: bool = True
    physical_surface_visibility_authoritative: bool = False
    schema: str = NESTED_TANGENT_PARENT_FRAME_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NESTED_TANGENT_PARENT_FRAME_SCHEMA:
            raise NestedTangentCompositingError(
                "invalid nested-tangent parent frame schema"
            )
        object.__setattr__(
            self,
            "mother_surface_id",
            _identity(self.mother_surface_id, "mother_surface_id"),
        )
        contacts = tuple(self.contacts)
        if len(contacts) != 2 or not all(
            isinstance(item, NestedTangentContactEvidence) for item in contacts
        ):
            raise NestedTangentCompositingError(
                "a nested tangent section requires exactly two contact records"
            )
        sphere_ids = tuple(item.sphere_surface_id for item in contacts)
        contact_source_ids = tuple(item.contact_source_id for item in contacts)
        sphere_item_ids = tuple(item.sphere_item_id for item in contacts)
        if sphere_ids != tuple(sorted(set(sphere_ids))):
            raise NestedTangentCompositingError(
                "nested contact records must name two unique sorted spheres"
            )
        if len(set(contact_source_ids)) != len(contact_source_ids):
            raise NestedTangentCompositingError(
                "nested contact source identities must be unique"
            )
        if len(set(sphere_item_ids)) != len(sphere_item_ids):
            raise NestedTangentCompositingError(
                "nested sphere painter identities must be unique"
            )
        if any(
            item.mother_surface_id != self.mother_surface_id
            for item in contacts
        ):
            raise NestedTangentCompositingError(
                "nested contact records must share the selected mother surface"
            )
        if {item.plane_position for item in contacts} != set(
            TangentSpherePlanePosition
        ):
            raise NestedTangentCompositingError(
                "nested contact records must occupy opposite plane sides"
            )
        object.__setattr__(self, "contacts", contacts)
        if not isinstance(self.sphere_pair_frame, GlobalQuadricFrame):
            raise TypeError("sphere_pair_frame must be a GlobalQuadricFrame")
        pair_surface_ids = tuple(
            item.surface_id
            for item in self.sphere_pair_frame.frame.surface_items
        )
        if pair_surface_ids != sphere_ids:
            raise NestedTangentCompositingError(
                "sphere-pair evidence must cover the two contact spheres exactly"
            )
        parent_ids = tuple(self.parent_item_ids)
        if len(parent_ids) != len(set(parent_ids)):
            raise NestedTangentCompositingError(
                "parent painter identities must be unique"
            )
        draw_order = tuple(self.draw_order)
        if len(draw_order) != len(set(draw_order)) or set(draw_order) != set(
            parent_ids
        ):
            raise NestedTangentCompositingError(
                "parent draw order must cover every painter item exactly once"
            )
        relations = tuple(self.order_relations)
        if not all(isinstance(item, QuadricPaintRelation) for item in relations):
            raise TypeError("order_relations must contain QuadricPaintRelation")
        relation_keys = tuple(
            (item.far_item_id, item.near_item_id, item.reason)
            for item in relations
        )
        if relation_keys != tuple(sorted(relation_keys)):
            raise NestedTangentCompositingError(
                "nested tangent painter relations must be sorted"
            )
        parent_set = set(parent_ids)
        if any(
            item.far_item_id not in parent_set
            or item.near_item_id not in parent_set
            for item in relations
        ):
            raise NestedTangentCompositingError(
                "nested tangent painter relation references a non-parent item"
            )
        rank = {item_id: index for index, item_id in enumerate(draw_order)}
        if any(
            rank[item.far_item_id] >= rank[item.near_item_id]
            for item in relations
        ):
            raise NestedTangentCompositingError(
                "nested tangent draw order violates a painter relation"
            )
        surface_map_entries: list[tuple[str, str]] = []
        for entry in self.surface_item_by_id:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise NestedTangentCompositingError(
                    "surface painter mapping entries must be identity pairs"
                )
            surface_map_entries.append(
                (
                    _identity(entry[0], "surface painter surface identity"),
                    _identity(entry[1], "surface painter item identity"),
                )
            )
        surface_map = tuple(surface_map_entries)
        if surface_map != tuple(sorted(surface_map)):
            raise NestedTangentCompositingError(
                "surface painter mapping must be sorted"
            )
        if len({key for key, _value in surface_map}) != len(surface_map):
            raise NestedTangentCompositingError(
                "surface painter mapping must have unique surface identities"
            )
        if len({value for _key, value in surface_map}) != len(surface_map):
            raise NestedTangentCompositingError(
                "surface painter mapping must have unique painter identities"
            )
        if any(value not in set(parent_ids) for _key, value in surface_map):
            raise NestedTangentCompositingError(
                "surface painter mapping references a non-parent item"
            )
        expected_surface_ids = {self.mother_surface_id, *sphere_ids}
        mapped_items = dict(surface_map)
        if set(mapped_items) != expected_surface_ids:
            raise NestedTangentCompositingError(
                "surface painter mapping must cover the mother and both spheres"
            )
        if any(
            mapped_items[item.sphere_surface_id] != item.sphere_item_id
            for item in contacts
        ):
            raise NestedTangentCompositingError(
                "sphere contact evidence disagrees with the surface painter mapping"
            )
        if (
            self.sphere_pair_frame.frame.paint_policy
            is not QuadricPaintPolicy.PHYSICAL
        ):
            raise NestedTangentCompositingError(
                "sphere-pair evidence must use physical paint policy"
            )
        relation_pairs = {
            (item.far_item_id, item.near_item_id) for item in relations
        }
        required_pair_relations = {
            (
                mapped_items[item.farther_surface_id],
                mapped_items[item.nearer_surface_id],
            )
            for item in self.sphere_pair_frame.surface_constraints
        }
        if not required_pair_relations.issubset(relation_pairs):
            raise NestedTangentCompositingError(
                "sphere-pair depth evidence is missing from parent relations"
            )
        if self.surface_layering_authoritative is not True:
            raise NestedTangentCompositingError(
                "registered nested parent layering must be authoritative"
            )
        if self.physical_surface_visibility_authoritative is not False:
            raise NestedTangentCompositingError(
                "teaching-transparent nested fills cannot claim optical authority"
            )
        object.__setattr__(self, "parent_item_ids", parent_ids)
        object.__setattr__(self, "order_relations", relations)
        object.__setattr__(self, "draw_order", draw_order)
        object.__setattr__(self, "surface_item_by_id", surface_map)

    @property
    def surface_items(self) -> dict[str, str]:
        return dict(self.surface_item_by_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "motherSurfaceId": self.mother_surface_id,
            "contacts": [item.to_dict() for item in self.contacts],
            "spherePairFrame": self.sphere_pair_frame.to_dict(),
            "parentItemIds": list(self.parent_item_ids),
            "orderRelations": [item.to_dict() for item in self.order_relations],
            "drawOrder": list(self.draw_order),
            "surfaceItemById": {
                key: value for key, value in self.surface_item_by_id
            },
            "surfaceLayeringAuthoritative": self.surface_layering_authoritative,
            "physicalSurfaceVisibilityAuthoritative": (
                self.physical_surface_visibility_authoritative
            ),
        }


def _merge_relations(
    relations: Sequence[QuadricPaintRelation],
) -> tuple[QuadricPaintRelation, ...]:
    reasons: dict[tuple[str, str], set[str]] = {}
    for relation in relations:
        if not isinstance(relation, QuadricPaintRelation):
            raise TypeError("relations must contain QuadricPaintRelation")
        pair = (relation.far_item_id, relation.near_item_id)
        reverse = (pair[1], pair[0])
        if reverse in reasons:
            raise NestedTangentCompositingError(
                "nested tangent painter relations contain contradictory evidence: "
                f"{pair[0]!r}, {pair[1]!r}"
            )
        reasons.setdefault(pair, set()).add(relation.reason)
    return tuple(
        QuadricPaintRelation(far, near, "+".join(sorted(values)))
        for (far, near), values in sorted(reasons.items())
    )


def _contact_residuals(
    mother: ConeSpec | CylinderSpec,
    sphere: SphereSpec,
    source: QuadricBoundarySource,
) -> tuple[float, float, float]:
    curve = source.curve
    if not isinstance(curve, EllipseArcCurve):
        raise NestedTangentCompositingError(
            "a nested tangent contact source must be an EllipseArcCurve"
        )
    if not curve.closed:
        raise NestedTangentCompositingError(
            "a nested tangent contact circle must cover one complete revolution"
        )
    first_length = float(np.linalg.norm(curve.first_axis))
    second_length = float(np.linalg.norm(curve.second_axis))
    length_scale = max(first_length, second_length)
    if abs(first_length - second_length) > 1.0e-10 * length_scale:
        raise NestedTangentCompositingError(
            "a nested tangent contact must be a circle, not a general ellipse"
        )

    sphere_center = np.asarray(sphere.center, dtype=float)
    sphere_residual = 0.0
    mother_residual = 0.0
    normal_residual = 0.0
    for parameter in np.linspace(
        curve.domain.start,
        curve.domain.end,
        17,
    )[:-1]:
        point = np.asarray(curve.point(float(parameter)), dtype=float)
        sphere_residual = max(
            sphere_residual,
            abs(float(np.linalg.norm(point - sphere_center)) - sphere.radius),
        )
        local = mother.frame.to_local_point(point)
        radial = float(np.linalg.norm(local[:2]))
        expected = (
            mother.radius
            if isinstance(mother, CylinderSpec)
            else abs(float(local[2])) * mother.slope
        )
        mother_residual = max(mother_residual, abs(radial - expected))

        first_gradient = sphere.support_quadric.gradient(point)
        second_gradient = mother.support_quadric.gradient(point)
        first_norm = float(np.linalg.norm(first_gradient))
        second_norm = float(np.linalg.norm(second_gradient))
        if first_norm <= 0.0 or second_norm <= 0.0:
            raise NestedTangentCompositingError(
                "a contact circle reached a singular surface normal"
            )
        normal_residual = max(
            normal_residual,
            float(
                np.linalg.norm(
                    np.cross(
                        first_gradient / first_norm,
                        second_gradient / second_norm,
                    )
                )
            ),
        )
    return sphere_residual, mother_residual, normal_residual


def _certify_contact(
    mother: ConeSpec | CylinderSpec,
    sphere: SphereSpec,
    plane: SectionPlane,
    source: QuadricBoundarySource,
    binding: NestedTangentSphereSpec,
    view: ParallelView,
    context: ResolvedGeometryContext,
) -> NestedTangentContactEvidence:
    if source.source_id != binding.contact_source_id:
        raise NestedTangentCompositingError(
            "contact source identity disagrees with its binding"
        )
    if source.owner_surface_id != sphere.surface_id:
        raise NestedTangentCompositingError(
            f"contact source {source.source_id!r} is not owned by sphere "
            f"{sphere.surface_id!r}"
        )
    if source.occlusion_scope not in {
        BoundaryOcclusionScope.OWNER_AND_EXTERNAL,
        BoundaryOcclusionScope.ALL_SURFACES,
    }:
        raise NestedTangentCompositingError(
            "contact-circle visibility must include its owner and mother surface"
        )

    local_center = mother.frame.to_local_point(sphere.center)
    lower, upper = mother.axial_range
    angular_tolerance = max(
        64.0 * context.epsilon(GeometryQuantity.ANGULAR),
        2.0e-9,
    )
    axial = float(local_center[2])
    expected_radius = (
        mother.radius
        if isinstance(mother, CylinderSpec)
        else abs(axial) * sin(mother.half_angle)
    )
    curve = source.curve
    contact_axis_scale = (
        max(
            float(np.linalg.norm(curve.first_axis)),
            float(np.linalg.norm(curve.second_axis)),
        )
        if isinstance(curve, EllipseArcCurve)
        else sphere.radius
    )
    contact_center = (
        curve.center if isinstance(curve, EllipseArcCurve) else sphere.center
    )
    length_tolerance = _local_boundary_tolerance(
        context,
        max(sphere.radius, expected_radius, contact_axis_scale),
        (
            *sphere.center,
            *contact_center,
            expected_radius,
        ),
    )
    model_radius_tolerance = _local_boundary_tolerance(
        context,
        max(sphere.radius, expected_radius),
    )
    expected_contact_radius = (
        mother.radius
        if isinstance(mother, CylinderSpec)
        else sphere.radius * abs(np.cos(mother.half_angle))
    )
    contact_radius_tolerance = _local_boundary_tolerance(
        context,
        max(contact_axis_scale, expected_contact_radius),
    )
    if isinstance(mother, ConeSpec):
        axial_radius_ulp = (
            32.0
            * abs(float(np.spacing(abs(axial))))
            * abs(sin(mother.half_angle))
        )
        length_tolerance = max(length_tolerance, axial_radius_ulp)
        model_radius_tolerance = max(
            model_radius_tolerance,
            axial_radius_ulp,
        )
    plane_tolerance = _point_pair_tolerance(
        context,
        sphere.radius,
        sphere.center,
        plane.point,
    )
    if float(np.linalg.norm(local_center[:2])) > length_tolerance:
        raise NestedTangentCompositingError(
            f"sphere {sphere.surface_id!r} is not coaxial with its mother surface"
        )
    sphere_lower = axial - sphere.radius
    sphere_upper = axial + sphere.radius
    lower_tolerance = _local_boundary_tolerance(
        context,
        sphere.radius,
        (sphere_lower, lower),
    )
    upper_tolerance = _local_boundary_tolerance(
        context,
        sphere.radius,
        (sphere_upper, upper),
    )
    if (
        sphere_lower < lower - lower_tolerance
        or sphere_upper > upper + upper_tolerance
    ):
        raise NestedTangentCompositingError(
            f"sphere {sphere.surface_id!r} exceeds the finite mother trim"
        )
    if abs(sphere.radius - expected_radius) > model_radius_tolerance:
        raise NestedTangentCompositingError(
            f"sphere {sphere.surface_id!r} is not internally tangent to its mother"
        )
    if isinstance(curve, EllipseArcCurve) and any(
        abs(float(np.linalg.norm(axis)) - expected_contact_radius)
        > contact_radius_tolerance
        for axis in (curve.first_axis, curve.second_axis)
    ):
        raise NestedTangentCompositingError(
            f"contact source {source.source_id!r} has the wrong contact-circle radius"
        )
    if isinstance(curve, EllipseArcCurve):
        center_delta = (
            np.asarray(curve.center, dtype=float)
            - np.asarray(sphere.center, dtype=float)
        )
        mother_axis = np.asarray(mother.frame.z_axis, dtype=float)
        actual_axial_delta = float(np.dot(center_delta, mother_axis))
        radial_delta = center_delta - actual_axial_delta * mother_axis
        expected_axial_delta = (
            0.0
            if isinstance(mother, CylinderSpec)
            else -axial * sin(mother.half_angle) ** 2
        )
        center_tolerance = _point_pair_tolerance(
            context,
            max(sphere.radius, expected_contact_radius),
            curve.center,
            sphere.center,
        )
        if isinstance(mother, ConeSpec):
            center_tolerance = max(
                center_tolerance,
                8.0
                * abs(float(np.spacing(abs(axial))))
                * sin(mother.half_angle) ** 2,
            )
        if (
            float(np.linalg.norm(radial_delta)) > center_tolerance
            or abs(actual_axial_delta - expected_axial_delta)
            > center_tolerance
        ):
            raise NestedTangentCompositingError(
                f"contact source {source.source_id!r} has the wrong contact-circle center"
            )

    signed_distance = plane.signed_distance(sphere.center)
    if abs(abs(signed_distance) - sphere.radius) > plane_tolerance:
        raise NestedTangentCompositingError(
            f"sphere {sphere.surface_id!r} is not tangent to the cutting plane"
        )
    denominator = float(
        np.dot(
            np.asarray(plane.normal, dtype=float),
            np.asarray(view.view_direction, dtype=float),
        )
    )
    if abs(denominator) <= angular_tolerance:
        raise NestedTangentCompositingError(
            "the cutting plane is edge-on to the parallel view"
        )
    plane_ray_parameter = -signed_distance / denominator
    if abs(plane_ray_parameter) <= context.epsilon(GeometryQuantity.DEPTH):
        raise NestedTangentCompositingError(
            f"sphere {sphere.surface_id!r} has unresolved plane depth"
        )

    sphere_residual, mother_residual, normal_residual = _contact_residuals(
        mother,
        sphere,
        source,
    )
    if max(sphere_residual, mother_residual) > length_tolerance:
        raise NestedTangentCompositingError(
            f"contact source {source.source_id!r} does not lie on both surfaces"
        )
    if normal_residual > angular_tolerance:
        raise NestedTangentCompositingError(
            f"contact source {source.source_id!r} is not a tangent contact"
        )
    return NestedTangentContactEvidence(
        sphere.surface_id,
        mother.surface_id,
        source.source_id,
        binding.sphere_item_id,
        float(signed_distance),
        float(plane_ray_parameter),
        (
            TangentSpherePlanePosition.PLANE_IN_FRONT
            if plane_ray_parameter > 0.0
            else TangentSpherePlanePosition.SPHERE_IN_FRONT
        ),
        sphere_residual,
        mother_residual,
        normal_residual,
    )


def _plane_relations(
    section: QuadricSectionCompositingFrame,
    contact: NestedTangentContactEvidence,
) -> tuple[QuadricPaintRelation, ...]:
    items = section.paint_items
    outlines = items.outline_by_role
    behind = (
        items.plane_behind,
        outlines[PlaneDepthRole.BEHIND_SURFACE],
    )
    behind_or_between = (
        *behind,
        items.plane_between,
        outlines[PlaneDepthRole.BETWEEN_SURFACE_SHEETS],
    )
    between_or_front = (
        items.plane_between,
        outlines[PlaneDepthRole.BETWEEN_SURFACE_SHEETS],
        items.plane_front,
        outlines[PlaneDepthRole.IN_FRONT_OF_SURFACE],
    )
    front = (
        items.plane_front,
        outlines[PlaneDepthRole.IN_FRONT_OF_SURFACE],
    )
    sphere = contact.sphere_item_id
    if contact.plane_is_in_front:
        return (
            *(
                QuadricPaintRelation(item_id, sphere, "plane_role_behind_sphere")
                for item_id in behind
            ),
            *(
                QuadricPaintRelation(sphere, item_id, "tangent_plane_in_front")
                for item_id in between_or_front
            ),
        )
    return (
        *(
            QuadricPaintRelation(item_id, sphere, "tangent_sphere_in_front")
            for item_id in behind_or_between
        ),
        *(
            QuadricPaintRelation(sphere, item_id, "plane_role_in_front_of_sphere")
            for item_id in front
        ),
    )


def _required_nested_parent_relations(
    section: QuadricSectionCompositingFrame,
    contacts: Sequence[NestedTangentContactEvidence],
    sphere_pair_frame: GlobalQuadricFrame,
    surface_item_by_id: Mapping[str, str],
) -> tuple[QuadricPaintRelation, ...]:
    """Return the mandatory parent skeleton for one registered nested frame."""

    relations: list[QuadricPaintRelation] = list(section.order_relations)
    for contact in contacts:
        relations.extend(
            (
                QuadricPaintRelation(
                    section.paint_items.surface_back,
                    contact.sphere_item_id,
                    "registered_nested_sphere_after_mother_back",
                ),
                QuadricPaintRelation(
                    contact.sphere_item_id,
                    section.paint_items.surface_front,
                    "registered_nested_sphere_before_mother_front",
                ),
                *_plane_relations(section, contact),
            )
        )
    try:
        relations.extend(
            QuadricPaintRelation(
                surface_item_by_id[item.farther_surface_id],
                surface_item_by_id[item.nearer_surface_id],
                "certified_sphere_pair_depth",
            )
            for item in sphere_pair_frame.surface_constraints
        )
    except KeyError as exc:
        raise NestedTangentCompositingError(
            "sphere-pair depth evidence references an unmapped surface"
        ) from exc
    return _merge_relations(relations)


def compute_nested_tangent_parent_frame(
    mother: ConeSpec | CylinderSpec,
    spheres: Sequence[SphereSpec],
    plane: SectionPlane,
    section_frame: QuadricSectionCompositingFrame,
    boundary_sources: Sequence[QuadricBoundarySource],
    bindings: Sequence[NestedTangentSphereSpec],
    view: ParallelView,
    *,
    context: ResolvedGeometryContext,
    max_chord_error: float,
    max_surface_segments: int,
) -> NestedTangentParentFrame:
    """Validate registered contacts and extend one section parent graph."""

    if not isinstance(mother, (ConeSpec, CylinderSpec)):
        raise TypeError("mother must be a ConeSpec or CylinderSpec")
    if not isinstance(section_frame, QuadricSectionCompositingFrame):
        raise TypeError("section_frame must be a QuadricSectionCompositingFrame")
    if section_frame.surface_id != mother.surface_id:
        raise NestedTangentCompositingError(
            "section frame does not belong to the registered mother surface"
        )
    if section_frame.plane != plane:
        raise NestedTangentCompositingError(
            "section frame does not belong to the supplied cutting plane"
        )
    section_visibility = section_frame.base_frame.visibility
    if (
        section_visibility.projection_matrix != view.projection_matrix
        or section_visibility.view_direction != view.view_direction
    ):
        raise NestedTangentCompositingError(
            "section frame does not belong to the supplied parallel view"
        )
    if not isinstance(context, ResolvedGeometryContext):
        raise TypeError("context must be a ResolvedGeometryContext")
    sphere_items = tuple(spheres)
    binding_items = tuple(
        sorted(bindings, key=lambda item: item.sphere_surface_id)
    )
    if len(sphere_items) != 2 or len(binding_items) != 2:
        raise NestedTangentCompositingError(
            "the nested tangent adapter currently requires exactly two spheres"
        )
    if not all(isinstance(item, SphereSpec) for item in sphere_items):
        raise TypeError("spheres must contain SphereSpec")
    by_sphere = {item.surface_id: item for item in sphere_items}
    by_binding = {item.sphere_surface_id: item for item in binding_items}
    if len(by_sphere) != 2 or set(by_binding) != set(by_sphere):
        raise NestedTangentCompositingError(
            "tangent bindings must cover both nested spheres exactly"
        )
    if any(item.mother_surface_id != mother.surface_id for item in binding_items):
        raise NestedTangentCompositingError(
            "every tangent binding must reference the selected mother surface"
        )
    source_map = {item.source_id: item for item in boundary_sources}
    if len(source_map) != len(tuple(boundary_sources)):
        raise NestedTangentCompositingError(
            "boundary source identities must be unique"
        )
    missing_sources = sorted(
        item.contact_source_id
        for item in binding_items
        if item.contact_source_id not in source_map
    )
    if missing_sources:
        raise NestedTangentCompositingError(
            "tangent bindings reference missing contact sources: "
            + ", ".join(missing_sources)
        )
    sphere_item_ids = tuple(item.sphere_item_id for item in binding_items)
    if len(set(sphere_item_ids)) != 2 or set(sphere_item_ids).intersection(
        section_frame.draw_order
    ):
        raise NestedTangentCompositingError(
            "sphere painter identities must be unique and separate from section items"
        )

    contacts = tuple(
        _certify_contact(
            mother,
            by_sphere[binding.sphere_surface_id],
            plane,
            source_map[binding.contact_source_id],
            binding,
            view,
            context,
        )
        for binding in sorted(binding_items, key=lambda item: item.sphere_surface_id)
    )
    plane_sides = {item.plane_position for item in contacts}
    if plane_sides != set(TangentSpherePlanePosition):
        raise NestedTangentCompositingError(
            "the two tangent spheres must lie on opposite sides of the plane"
        )

    sphere_pair_frame = compute_global_quadric_frame(
        (),
        sphere_items,
        view,
        context=context,
        paint_policy=QuadricPaintPolicy.PHYSICAL,
        max_chord_error=max_chord_error,
        max_segments=max_surface_segments,
    )
    item_by_surface = {
        item.sphere_surface_id: item.sphere_item_id for item in binding_items
    }
    normalized = _required_nested_parent_relations(
        section_frame,
        contacts,
        sphere_pair_frame,
        item_by_surface,
    )
    nodes = (*section_frame.draw_order, *sphere_item_ids)
    preferred = {
        item_id: index for index, item_id in enumerate(nodes)
    }
    draw_order = stable_topological_sort(
        nodes,
        tuple(
            PainterConstraint(item.far_item_id, item.near_item_id)
            for item in normalized
        ),
        key=lambda item_id: (preferred.get(item_id, len(preferred)), item_id),
    )
    return NestedTangentParentFrame(
        mother.surface_id,
        contacts,
        sphere_pair_frame,
        tuple(nodes),
        normalized,
        draw_order,
        tuple(
            sorted(
                {
                    mother.surface_id: section_frame.paint_items.surface_front,
                    **item_by_surface,
                }.items()
            )
        ),
    )


__all__ = [
    "NESTED_TANGENT_CONTACT_EVIDENCE_SCHEMA",
    "NESTED_TANGENT_PARENT_FRAME_SCHEMA",
    "NestedTangentCompositingError",
    "NestedTangentContactEvidence",
    "NestedTangentParentFrame",
    "NestedTangentSphereSpec",
    "TangentSpherePlanePosition",
    "compute_nested_tangent_parent_frame",
]
