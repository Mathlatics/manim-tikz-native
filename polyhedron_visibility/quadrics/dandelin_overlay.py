"""Renderer-neutral teaching-overlay bundle for Dandelin constructions.

This bundle is deliberately diagrammatic.  It groups already-certified
Dandelin geometry into a stable display order, but it does not claim that a
contained sphere and its tangent cone have been physically depth-composited.
That distinction is persisted as ``visibility_authoritative=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Literal

from .contract import PlaneDisplayPatchSpec, SphereSpec
from .curves import CircleArcCurve, SegmentCurve
from .dandelin import ContextInput, DandelinConstruction3D
from .planar_curves import PlanarPoint3D


DANDELIN_TEACHING_OVERLAY_SCHEMA = "manim-dandelin-teaching-overlay-3d/v1"


class DandelinTeachingOverlayError(ValueError):
    """A diagrammatic Dandelin overlay was configured inconsistently."""


@dataclass(frozen=True, slots=True)
class DandelinTeachingOverlay3D:
    """Stable geometry bundle for a non-authoritative classroom overlay."""

    construction: DandelinConstruction3D
    directrix_curves: tuple[SegmentCurve, ...]
    draw_order: tuple[str, ...]
    mode: Literal["diagrammatic"] = "diagrammatic"
    visibility_authoritative: bool = False
    schema: str = DANDELIN_TEACHING_OVERLAY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != DANDELIN_TEACHING_OVERLAY_SCHEMA:
            raise DandelinTeachingOverlayError(
                "invalid Dandelin teaching-overlay schema"
            )
        if not isinstance(self.construction, DandelinConstruction3D):
            raise TypeError("construction must be a DandelinConstruction3D")
        if self.mode != "diagrammatic":
            raise DandelinTeachingOverlayError(
                "Dandelin v1 supports only a diagrammatic teaching overlay"
            )
        if self.visibility_authoritative is not False:
            raise DandelinTeachingOverlayError(
                "Dandelin teaching overlays must remain explicitly non-authoritative"
            )
        curves = tuple(self.directrix_curves)
        if not all(isinstance(item, SegmentCurve) for item in curves):
            raise TypeError("directrix_curves must contain SegmentCurve values")
        if tuple(sorted(curves, key=lambda item: item.curve_id)) != curves:
            raise DandelinTeachingOverlayError(
                "directrix curves must use canonical identity order"
            )
        order = tuple(self.draw_order)
        expected = self._expected_draw_order(curves)
        if order != expected:
            raise DandelinTeachingOverlayError(
                "teaching-overlay draw_order is not the canonical diagrammatic order"
            )
        object.__setattr__(self, "directrix_curves", curves)
        object.__setattr__(self, "draw_order", order)

    def _expected_draw_order(
        self,
        directrix_curves: tuple[SegmentCurve, ...],
    ) -> tuple[str, ...]:
        return (
            *(f"overlay:surface:{item.surface_id}" for item in self.sphere_surfaces),
            *(item.curve_id for item in self.contact_curves),
            *(item.curve_id for item in directrix_curves),
            *(f"overlay:focus:{item.focus_id}" for item in self.construction.spheres),
        )

    @property
    def sphere_surfaces(self) -> tuple[SphereSpec, ...]:
        return self.construction.sphere_surfaces

    @property
    def contact_curves(self) -> tuple[CircleArcCurve, ...]:
        return tuple(
            item.lower_to_analytic_curve()
            for item in self.construction.cone_contact_circles
        )

    @property
    def focus_points(self) -> tuple[PlanarPoint3D, ...]:
        return self.construction.focus_points

    @property
    def curves(self) -> tuple[CircleArcCurve | SegmentCurve, ...]:
        return (*self.contact_curves, *self.directrix_curves)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "constructionId": self.construction.construction_id,
            "mode": self.mode,
            "visibilityAuthoritative": self.visibility_authoritative,
            "sphereIds": [item.surface_id for item in self.sphere_surfaces],
            "contactCurveIds": [item.curve_id for item in self.contact_curves],
            "directrixCurveIds": [item.curve_id for item in self.directrix_curves],
            "focusIds": [item.focus_id for item in self.construction.spheres],
            "drawOrder": list(self.draw_order),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def build_dandelin_teaching_overlay(
    construction: DandelinConstruction3D,
    patch: PlaneDisplayPatchSpec,
    *,
    context: ContextInput = None,
    mode: str = "diagrammatic",
) -> DandelinTeachingOverlay3D:
    """Bundle one construction for an explicitly diagrammatic renderer."""

    if not isinstance(construction, DandelinConstruction3D):
        raise TypeError("construction must be a DandelinConstruction3D")
    if not isinstance(patch, PlaneDisplayPatchSpec):
        raise TypeError("patch must be a PlaneDisplayPatchSpec")
    if mode != "diagrammatic":
        raise DandelinTeachingOverlayError(
            "physical and depth-aware contained-surface compositing is not part "
            "of the Dandelin v1 overlay contract"
        )
    directrices = tuple(
        sorted(
            construction.directrix_segments(patch, context=context),
            key=lambda item: item.curve_id,
        )
    )
    draw_order = (
        *(f"overlay:surface:{item.surface_id}" for item in construction.sphere_surfaces),
        *(
            item.lower_to_analytic_curve().curve_id
            for item in construction.cone_contact_circles
        ),
        *(item.curve_id for item in directrices),
        *(f"overlay:focus:{item.focus_id}" for item in construction.spheres),
    )
    return DandelinTeachingOverlay3D(
        construction=construction,
        directrix_curves=directrices,
        draw_order=draw_order,
    )


def canonical_dandelin_teaching_overlay_json(
    overlay: DandelinTeachingOverlay3D,
) -> str:
    if not isinstance(overlay, DandelinTeachingOverlay3D):
        raise TypeError("overlay must be a DandelinTeachingOverlay3D")
    return overlay.canonical_json()


__all__ = [
    "DANDELIN_TEACHING_OVERLAY_SCHEMA",
    "DandelinTeachingOverlay3D",
    "DandelinTeachingOverlayError",
    "build_dandelin_teaching_overlay",
    "canonical_dandelin_teaching_overlay_json",
]
